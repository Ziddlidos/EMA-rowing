#include <SoftwareSerial.h>

SoftwareSerial BTSerial(10, 11); // RX | TX
String flag;
int sobe = 7; // utilizando um botão de dois terminais nesse pino
int desce = 6;
int acaba = 5;


void setup()
{
  pinMode(sobe, INPUT);
  pinMode(desce, INPUT);
  pinMode(acaba, INPUT);
  digitalWrite(sobe, HIGH);
  digitalWrite(desce, HIGH);


  pinMode(13, OUTPUT);    // Vamos usar LED onboard como sinalizador de comunicação
  BTSerial.begin(9600);
  Serial.begin(9600);

  inicializacao();

}

void loop()
{
  /*
    if (digitalRead(sobe) == LOW) {
    // Le o que foi digitado no Monitor Serial do Arduino e envie ao HC-05
    BTSerial.write("c");
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(1);
    digitalWrite(sobe, HIGH);
    delay(200);
    }
  */
  delay(1000);
}

void inicializacao() {

  int corrente = 10;
  int pw = 200;
  int freq = 50;
  int mode = 2;
  bool conexao = false;

  //verificando conexão
  Serial.println("Conectando...");
  while (conexao == false) {
    if (BTSerial.available()) {
      flag = BTSerial.readString();     
      Serial.println(flag);
      if (flag.equals("a"))  {
        conexao = true;
        Serial.println("entrou!");
      }
    }

  }

  BTSerial.write(1);

  Serial.println("Conectado!");
  delay(1000);
  // pegando parametros de quantidade de canais, corrente e largura de pulso (pw)
  //corrente
  while (digitalRead(acaba) == LOW) {

    Serial.print("Corrente:");
    Serial.print("\t");
    Serial.println(corrente);

    if (digitalRead(sobe) == LOW) {
      digitalWrite(sobe, HIGH);
      corrente = corrente + 2;
      delay(100);
    }
    else if (digitalRead(desce) == LOW) {
      digitalWrite(desce, HIGH);
      corrente = corrente - 2;
      delay(100);
    }
  }
  delay(200);
  //largura de pulso
  while (digitalRead(acaba) == LOW) {

    Serial.print("Largura de Pulso::");
    Serial.print("\t");
    Serial.println(pw);

    if (digitalRead(sobe) == LOW) {
      digitalWrite(sobe, HIGH);
      pw = pw + 2;
      delay(100);
    }
    else if (digitalRead(desce) == LOW) {
      digitalWrite(desce, HIGH);
      pw = pw - 2;
      delay(100);
    }
  }
  delay(200);
  //Frequencia
  while (digitalRead(acaba) == LOW) {
    Serial.print("Frequencia:");
    Serial.print("\t");
    Serial.println(freq);

    if (digitalRead(sobe) == LOW) {
      digitalWrite(sobe, HIGH);
      freq = freq + 2;
      delay(100);
    }
    else if (digitalRead(desce) == LOW) {
      digitalWrite(desce, HIGH);
      freq = freq - 2;
      delay(100);
    }
  }
  delay(200);
  //modo de operação
  while (digitalRead(acaba) == LOW) {
    Serial.print("Qtd de Canais:");
    Serial.print("\t");
    Serial.println(mode);

    if (digitalRead(sobe) == LOW) {
      digitalWrite(sobe, HIGH);
      mode = mode + 2;
      delay(100);
    }
    else if (digitalRead(desce) == LOW) {
      digitalWrite(desce, HIGH);
      mode = mode - 2;
      delay(100);
    }
  }
  delay(200);

  // enviando dados pela serial (bluetooth)
  BTSerial.write("c");//marcador de corrente

  if (qtdAlgarismos(corrente) == 3) {
    BTSerial.print(corrente);
  } else if (qtdAlgarismos(corrente) == 2) {
    BTSerial.print(0);
    BTSerial.print(corrente);
  } else if (qtdAlgarismos(corrente) == 1) {
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(corrente);
  }

  BTSerial.write("p");//marcador de largura de pulso

  if (qtdAlgarismos(pw) == 3) {
    BTSerial.print(pw);
  } else if (qtdAlgarismos(pw) == 2) {
    BTSerial.print(0);
    BTSerial.print(pw);
  } else if (qtdAlgarismos(pw) == 1) {
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(pw);
  }

  BTSerial.write("f");//marcador de frequecia

  if (qtdAlgarismos(freq) == 3) {
    BTSerial.print(freq);
  } else if (qtdAlgarismos(freq) == 2) {
    BTSerial.print(0);
    BTSerial.print(freq);
  } else if (qtdAlgarismos(freq) == 1) {
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(freq);
  }

  BTSerial.write("m");//marcador do modo

  if (qtdAlgarismos(mode) == 3) {
    BTSerial.print(mode);
  } else if (qtdAlgarismos(mode) == 2) {
    BTSerial.print(0);
    BTSerial.print(mode);
  } else if (qtdAlgarismos(mode) == 1) {
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(mode);
  }



}
int qtdAlgarismos(int numero) {
  int cont = 0;
  while (numero != 0) {
    // n = n/10
    numero /= 10;
    cont++;
  }
  return cont;
}
