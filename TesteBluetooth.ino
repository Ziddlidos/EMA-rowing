#include <SoftwareSerial.h>

SoftwareSerial BTSerial(10, 11); // RX | TX
int flag = 1;
int flag1 = 0;
bool conexao = false;
void setup()
{
  pinMode(13, OUTPUT);    // Vamos usar LED onboard como sinalizador de comunicação
  BTSerial.begin(9600); 
  Serial.begin(9600);
}

void loop()
{

  // Le a saída do HC-05 and envie ao Monitor Serial do Arduino 
  if (BTSerial.available()){
    digitalWrite(13, HIGH);
    Serial.write(BTSerial.read());
    digitalWrite(13, LOW);
  }

  // Le o que foi digitado no Monitor Serial do Arduino e envie ao HC-05
  if (Serial.available()){
    digitalWrite(13, HIGH);
    BTSerial.write(Serial.read());
    digitalWrite(13, LOW);
  }
}

/*
int init(){
  //verificando conexão
  print("Conectando...");
  while (conexao == false){
      BTserial.write(flag);
      if(BTSerial.available()){
         flag1 = BTSerial.read();
        }
      if (flag1 == flag)  {
        conexao = true;
        }
    }
    print("Conectado!");
    delay(1000) ;
    BTserial.write("a");
  // pegando parametros de quantidade de canais, corrente e largura de pulso (pw)
  
  }
*/
