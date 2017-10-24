#include <SoftwareSerial.h>
#include <LiquidCrystal.h>


LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
SoftwareSerial BTSerial(9, 10); // RX | TX

String flag;
int sobe = 8; // utilizando um botão de dois terminais nesse pino
int desce = 7;
int acaba = 6;

bool stim = true;

void setup()
{
  lcd.begin(16, 2);

  pinMode(sobe, INPUT);
  pinMode(desce, INPUT);
  pinMode(acaba, INPUT);
  digitalWrite(sobe, HIGH);
  digitalWrite(desce, HIGH);
  digitalWrite(acaba, HIGH);

  BTSerial.begin(9600);

  inicializacao();

  bool conexao = false;
  while (conexao == false) {
    if (BTSerial.available()) {
      flag = BTSerial.readString();
      lcd.print(flag);
      if (flag.equals("a"))  {
        conexao = true;
        lcd.clear();
        lcd.print("Iniciando estimulacao");
      }
    }
  }
}

void loop()
{
  int state0 = 0;
  int state1 = 1;
  while (stim) {
    if (digitalRead(sobe) == LOW) {
      digitalWrite(sobe, HIGH);
      state0 = 1;
      lcd.clear();
      lcd.print("Extensao");
    }
    else if (digitalRead(desce) == LOW) {
      digitalWrite(desce, HIGH);
      state0 = 2;
      lcd.clear();
      lcd.print("Contracao");
    }
    else if (digitalRead(acaba) == LOW) {
      digitalWrite(acaba, HIGH);
      state0 = 3;
      
      lcd.clear();
      lcd.print("Fim");
    }
    else {
      state0 = 0;
    }
    if (state1 != state0) {
      BTSerial.print(state0);
      state1 = state0;
    }
    if (state0 == 3) {
      BTSerial.print(state0);
      stim = false;
    }
    delay(30);
  }
}

void inicializacao() {

  int corrente = 10;
  int pw = 200;
  int freq = 50;
  int mode = 2;
  bool conexao = false;

  //verificando conexão
  lcd.clear();
  lcd.print("Conectando...");
  while (conexao == false) {
    if (BTSerial.available()) {
      flag = BTSerial.readString();
      if (flag.equals("a"))  {
        conexao = true;
      }
    }

  }

  BTSerial.write(1);
  lcd.clear();
  lcd.print("Conectado!");
  delay(1000);
  // pegando parametros de quantidade de canais, corrente e largura de pulso (pw)
  //corrente
  lcd.clear();
  lcd.print("Corrente:");
  while (digitalRead(acaba) == HIGH) {

    lcd.setCursor(0, 1);
    lcd.println(corrente);

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

  delay(500);
  //largura de pulso
  lcd.clear();
  lcd.print("Largura de Pulso:");
  while (digitalRead(acaba) == HIGH) {

    lcd.setCursor(0, 1);
    lcd.println(pw);

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
  digitalWrite(acaba, HIGH);
  delay(500);
  //Frequencia

  lcd.clear();
  lcd.print("Frequencia:");
  while (digitalRead(acaba) == HIGH) {

    lcd.setCursor(0, 1);
    lcd.println(freq);

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
  digitalWrite(acaba, HIGH);
  delay(500);
  //modo de operação

  lcd.clear();
  lcd.print("Qtd de Canais:");
  while (digitalRead(acaba) == HIGH) {

    lcd.setCursor(0, 1);
    lcd.print(mode);

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
  digitalWrite(acaba, HIGH);
  delay(500);

  // enviando dados pela serial (bluetooth)
  BTSerial.print("c");//marcador de corrente

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

  BTSerial.print("p");//marcador de largura de pulso

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

  BTSerial.print("f");//marcador de frequecia

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

  BTSerial.print("m");//marcador do modo

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
  lcd.clear();
  lcd.print("Enviando");


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
