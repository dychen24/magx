#include "Adafruit_MLX90393.h"
#define num 8

Adafruit_MLX90393 sensor[num];
int CS[] = {16, 15, 7, 11, 26, 25, 27, 30};

// int CS[num] = {16, 15, 7, 11};

float x[num];
float y[num];
float z[num];

void setup()
{
  dwt_enable(); // For more accurate micros() on Feather
  Serial.begin(115200);
  /* Wait for serial on USB platforms. */
  pinMode(LED_BUILTIN, OUTPUT); // Indicator of whether the sensors are all found
  digitalWrite(LED_BUILTIN, LOW);
  while (!Serial)
  {
    delayMicroseconds(10);
  }
  delayMicroseconds(1000);
  for (int i = 0; i < num; ++i)
  {
    sensor[i] = Adafruit_MLX90393();
    while (!sensor[i].begin_SPI(CS[i]))
    {
      Serial.print("No sensor ");
      Serial.print(i + 1);
      Serial.println(" found ... check your wiring?");
      delayMicroseconds(500);
    }
    Serial.print("Sensor ");
    Serial.print(i + 1);
    Serial.println(" found!");

    while (!sensor[i].setOversampling(MLX90393_OSR_3))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" reset OSR!");
      delayMicroseconds(500);
    }
    delayMicroseconds(500);
    while (!sensor[i].setFilter(MLX90393_FILTER_5))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" reset filter!");
      delayMicroseconds(500);
    }
  }
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop()
{
  int start_time = micros();
  for (int i = 0; i < num; ++i)
  {
    sensor[i].startSingleMeasurement();
    //delayMicroseconds(50);
  }
  //delayMicroseconds(mlx90393_tconv[4][2]*1000-4000);
  delayMicroseconds(mlx90393_tconv[5][3] * 1000 - 200);
  Serial.println("###################");
  for (int i = 0; i < num; ++i)
  {
    if (!sensor[i].readMeasurement(&x[i], &y[i], &z[i]))
    {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" no data read!");
      digitalWrite(LED_BUILTIN, LOW);
    }
    Serial.print("Sensor ");
    Serial.print(i + 1);
    Serial.print(" : ");
    Serial.print(x[i]);
    Serial.print(", ");
    Serial.print(y[i]);
    Serial.print(", ");
    Serial.println(z[i]);
    //delayMicroseconds(50);
  }
  //delayMicroseconds(500);
  int elapsed_time = micros() - start_time;
  Serial.print(1000000 / elapsed_time);
  Serial.println(" Hz");
}
