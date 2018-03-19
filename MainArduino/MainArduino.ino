//teste

#include <SoftwareSerial.h>
#include <LiquidCrystal.h>


LiquidCrystal lcd(7, 6, 5, 4, 3, 2);
SoftwareSerial BTSerial(8, 9); // RX | TX

String flag;
int sobe = 10; // utilizando um botão de dois terminais nesse pino
int desce = 12;
int acaba = 11;

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

  BTSerial.begin(115200);

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

  int corrente_quad = 10;
  int corrente_isq = 10;
  int pw = 200;
  int freq = 50;
  int mode = 2;
  bool conexao = false;
  int upBTN = HIGH;
  int downBTN = HIGH;
  int lastReadingUp = HIGH;
  int lastReadingDown = HIGH;
  long lastSwitchTimeUp = 0;
  long switchTimeUp = 500;
  long lastSwitchTimeDown = 0;
  long switchTimeDown = 500;
  long longSwitchTime = 500;
  long shortSwitchTime = 100;

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
  //corrente1 - quadriceps
  lcd.clear();
  lcd.print("Corrente Quad.:");
  while (digitalRead(acaba) == HIGH) {

    
    lcd.setCursor(0, 1);
    lcd.print(corrente_quad);
    lcd.print("     ");

    upBTN = digitalRead(sobe);
    downBTN = digitalRead(desce);
    //delay(100);
    if (upBTN == LOW && ((((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) || lastReadingUp == HIGH)) {
      //digitalWrite(sobe, HIGH);
      if (((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) {
        switchTimeUp = shortSwitchTime;
      }
      lastSwitchTimeUp = millis();
      lastReadingUp = upBTN;
      corrente_quad = corrente_quad + 2;
      //delay(100);
    }
    if (upBTN == HIGH) {
      lastReadingUp = upBTN;
      switchTimeUp = longSwitchTime;
      lastSwitchTimeUp = 0;
    }
    if (downBTN == LOW && ((((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) || lastReadingDown == HIGH)) {
      //digitalWrite(desce, HIGH);
      if (((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) {
        switchTimeDown = shortSwitchTime;
      }
      lastSwitchTimeDown = millis();
      lastReadingDown = downBTN;
      corrente_quad = corrente_quad - 2;
      if (corrente_quad < 0) {
        corrente_quad = 0;
      }
      //delay(100);      
    }
    if (downBTN == HIGH) {
      lastReadingDown = downBTN;
      switchTimeDown = longSwitchTime;
      lastSwitchTimeDown = 0;
    }
  }

  delay(500);
  //corrente2 - isquiotibiais
  lcd.clear();
  lcd.print("Corrente Isq.:");
  while (digitalRead(acaba) == HIGH) {

    
    lcd.setCursor(0, 1);
    lcd.print(corrente_isq);
    lcd.print("     ");

    upBTN = digitalRead(sobe);
    downBTN = digitalRead(desce);
    //delay(100);
    if (upBTN == LOW && ((((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) || lastReadingUp == HIGH)) {
      //digitalWrite(sobe, HIGH);
      if (((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) {
        switchTimeUp = shortSwitchTime;
      }
      lastSwitchTimeUp = millis();
      lastReadingUp = upBTN;
      corrente_isq = corrente_isq + 2;
      //delay(100);
    }
    if (upBTN == HIGH) {
      lastReadingUp = upBTN;
      switchTimeUp = longSwitchTime;
      lastSwitchTimeUp = 0;
    }
    if (downBTN == LOW && ((((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) || lastReadingDown == HIGH)) {
      //digitalWrite(desce, HIGH);
      if (((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) {
        switchTimeDown = shortSwitchTime;
      }
      lastSwitchTimeDown = millis();
      lastReadingDown = downBTN;
      corrente_isq = corrente_isq - 2;
      if (corrente_isq < 0) {
        corrente_isq = 0;
      }
      //delay(100);      
    }
    if (downBTN == HIGH) {
      lastReadingDown = downBTN;
      switchTimeDown = longSwitchTime;
      lastSwitchTimeDown = 0;
    }
  }

  delay(500);
  //largura de pulso
  lcd.clear();
  lcd.print("Largura de Pulso:");  
  while (digitalRead(acaba) == HIGH) {
    
    lcd.setCursor(0, 1);
    lcd.print(pw);

    upBTN = digitalRead(sobe);
    downBTN = digitalRead(desce);
    if (upBTN == LOW && ((((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) || lastReadingUp == HIGH)) {
      //digitalWrite(sobe, HIGH);
      if (((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) {
        switchTimeUp = shortSwitchTime;
      }
      lastSwitchTimeUp = millis();
      lastReadingUp = upBTN;
      pw = pw + 10;
      //delay(100);
    }
    if (upBTN == HIGH) {
      lastReadingUp = upBTN;
      switchTimeUp = longSwitchTime;
      lastSwitchTimeUp = 0;
    }
    if (downBTN == LOW && ((((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) || lastReadingDown == HIGH)) {
      //digitalWrite(desce, HIGH);
      if (((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) {
        switchTimeDown = shortSwitchTime;
      }
      lastSwitchTimeDown = millis();
      lastReadingDown = downBTN;
      pw = pw - 10;
      if (pw < 0) {
        pw = 0;
      }
      //delay(100);
    }
    if (downBTN == HIGH) {
      lastReadingDown = downBTN;
      switchTimeDown = longSwitchTime;
      lastSwitchTimeDown = 0;
    }
  }
  //digitalWrite(acaba, HIGH);
  delay(500);
  //Frequencia

  lcd.clear();
  lcd.print("Frequencia:");  
  while (digitalRead(acaba) == HIGH) {

    lcd.setCursor(0, 1);
    lcd.print(freq);

    upBTN = digitalRead(sobe);
    downBTN = digitalRead(desce);
    if (upBTN == LOW && ((((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) || lastReadingUp == HIGH)) {
      //digitalWrite(sobe, HIGH);
      if (((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) {
        switchTimeUp = shortSwitchTime;
      }
      lastSwitchTimeUp = millis();
      lastReadingUp = upBTN;
      freq = freq + 5;
      //delay(100);
    }
    if (upBTN == HIGH) {
      lastReadingUp = upBTN;
      switchTimeUp = longSwitchTime;
      lastSwitchTimeUp = 0;
    }
    if (downBTN == LOW && ((((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) || lastReadingDown == HIGH)) {
      //digitalWrite(desce, HIGH);
      if (((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) {
        switchTimeDown = shortSwitchTime;
      }
      lastSwitchTimeDown = millis();
      lastReadingDown = downBTN;
      freq = freq - 5;
      if (freq < 0) {
        freq = 0;
      }
      //delay(100);
    }
    if (downBTN == HIGH) {
      lastReadingDown = downBTN;
      switchTimeDown = longSwitchTime;
      lastSwitchTimeDown = 0;
    }
  }
  //digitalWrite(acaba, HIGH);
  delay(500);
  //modo de operação

  lcd.clear();
  lcd.print("Qtd de Canais:");
  while (digitalRead(acaba) == HIGH) {
    
    lcd.setCursor(0, 1);
    lcd.print(mode);

    upBTN = digitalRead(sobe);
    downBTN = digitalRead(desce);
    if (upBTN == LOW && ((((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) || lastReadingUp == HIGH)) {
      //digitalWrite(sobe, HIGH);
      if (((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) {
        switchTimeUp = shortSwitchTime;
      }
      lastSwitchTimeUp = millis();
      lastReadingUp = upBTN;
      mode = mode + 2;
      //delay(100);
    }
    if (downBTN == HIGH) {
      lastReadingDown = downBTN;
      switchTimeDown = longSwitchTime;
      lastSwitchTimeDown = 0;
    }
    if (downBTN == LOW && ((((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) || lastReadingDown == HIGH)) {
      //digitalWrite(desce, HIGH);
      if (((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) {
        switchTimeDown = shortSwitchTime;
      }
      lastSwitchTimeDown = millis();
      lastReadingDown = downBTN;      
      mode = mode - 2;
      if (mode < 0) {
        mode = 0;
      }
      //delay(100);
    }
    if (upBTN == HIGH) {
      lastReadingUp = upBTN;
      switchTimeUp = longSwitchTime;
      lastSwitchTimeUp = 0;
    }
  }
  //digitalWrite(acaba, HIGH);
  delay(500);

  // enviando dados pela serial (bluetooth)
  BTSerial.print("c");//marcador de corrente_quad

  if (qtdAlgarismos(corrente_quad) == 3) {
    BTSerial.print(corrente_quad);
  } else if (qtdAlgarismos(corrente_quad) == 2) {
    BTSerial.print(0);
    BTSerial.print(corrente_quad);
  } else if (qtdAlgarismos(corrente_quad) == 1) {
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(corrente_quad);
  }
  
  BTSerial.print("x");//marcador de corrente_isq

  if (qtdAlgarismos(corrente_isq) == 3) {
    BTSerial.print(corrente_isq);
  } else if (qtdAlgarismos(corrente_isq) == 2) {
    BTSerial.print(0);
    BTSerial.print(corrente_isq);
  } else if (qtdAlgarismos(corrente_isq) == 1) {
    BTSerial.print(0);
    BTSerial.print(0);
    BTSerial.print(corrente_isq);
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
