//Arduino Code

#include <ArduinoBLE.h>
#include <PDM.h>
#include <Mosquito_Detection_FYP1_inferencing.h>
#include <DHT.h>

#define DHTPIN A4
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);

BLEService sensorService("180A");
BLEFloatCharacteristic temperatureCharacteristic("2A6E", BLERead | BLENotify);
BLEFloatCharacteristic humidityCharacteristic("2A6F", BLERead | BLENotify);
BLEStringCharacteristic classificationCharacteristic("1a3f58e4-2f49-4df5-b2e0-0ecb0e2f6a42", BLERead | BLENotify, 32);

typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("Mosquito WingBeat Sound Classification");

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        Serial.println("ERR: Failed to setup audio sampling");
        return;
    }

    dht.begin();

    if (!BLE.begin()) {
        Serial.println("Starting BLE failed!");
        while (1);
    }

    BLE.setLocalName("Nano33BLE");
    BLE.setAdvertisedService(sensorService);
    sensorService.addCharacteristic(temperatureCharacteristic);
    sensorService.addCharacteristic(humidityCharacteristic);
    sensorService.addCharacteristic(classificationCharacteristic);
    BLE.addService(sensorService);
    temperatureCharacteristic.writeValue(0.0);
    humidityCharacteristic.writeValue(0.0);
    classificationCharacteristic.writeValue("");
    BLE.advertise();
    Serial.println("BLE Peripheral device is now advertising");
}

void loop() {
    BLEDevice central = BLE.central();

    if (central) {
        Serial.print("Connected to central: ");
        Serial.println(central.address());

        while (central.connected()) {
            float humidity = dht.readHumidity();
            float temperature = dht.readTemperature();

            if (isnan(humidity) || isnan(temperature)) {
                Serial.println("Failed to read from DHT sensor!");
            } else {
                temperatureCharacteristic.writeValue(temperature);
                humidityCharacteristic.writeValue(humidity);

                Serial.print("Humidity: ");
                Serial.print(humidity);
                Serial.print(" %\t");
                Serial.print("Temperature: ");
                Serial.print(temperature);
                Serial.println(" *C");
            }

            delay(2000);
            Serial.println("Recording...");
            if (!microphone_inference_record()) {
                Serial.println("ERR: Failed to record audio...");
                return;
            }

            signal_t signal;
            signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
            signal.get_data = &microphone_audio_signal_get_data;
            ei_impulse_result_t result = { 0 };

            EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
            if (r != EI_IMPULSE_OK) {
                Serial.print("ERR: Failed to run classifier (");
                Serial.print(r);
                Serial.println(")");
                return;
            }

            int maxIndex = 0;
            float maxValue = -1.0;
            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                if (result.classification[ix].value > maxValue) {
                    maxValue = result.classification[ix].value;
                    maxIndex = ix;
                }
            }

            int mosquitoTypeCode = maxIndex + 1;
            String mosquitoTypeString = String(mosquitoTypeCode);
            classificationCharacteristic.writeValue(mosquitoTypeString);

            Serial.print("Detected Mosquito Type Code: ");
            Serial.println(mosquitoTypeCode);

            delay(2000);
        }

        Serial.print("Disconnected from central: ");
        Serial.println(central.address());
    }
}

void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };
    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);
    if (r > 0) {
        Serial.write(print_buf);
    }
}

static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);
    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];
            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if(inference.buffer == NULL) {
        return false;
    }
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;
    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        Serial.println("Failed to start PDM!");
        microphone_inference_end();
        return false;
    }
    PDM.setGain(127);
    return true;
}

static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;
    while(inference.buf_ready == 0) {
        delay(10);
    }
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif