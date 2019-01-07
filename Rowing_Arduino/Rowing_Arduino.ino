#include <LiquidCrystal.h>
#include <LiquidCrystal.h>
#include <SoftwareSerial.h>
LiquidCrystal lcd(9, 7, 8, 12, 10, 11);
#define LED_Alerta 6 // Pin is connected to the LED ALERTA
#define LED_Direito 5 // Pin is connected to the LED DIREITO
#define LED_Esquerdo 13 // Pin is connected to the LED ESQUERDO
/**
* ponteiro de função da máquina de estados. Ele aponta sempre para a função da
* máquina de estados que deve ser executada */
bool aux_CH12 = false;
bool aux_CH34 = false;
bool config = true;
bool print;
bool run = false;
bool stim = true;
byte channels = B00000000;
byte set_channels = B00000000;
char acaba = 3;
char corrente_CH12 = 10;
char corrente_CH34 = 6;
char corrente_CH56 = 10;
char corrente_CH78 = 6;
char desce = 4;
char freq = 40;
char passo;
char sobe = 2;
int mode = 0;
int pw = 300;
int variavel;
String flag;
String text;
String unidade;
void(* PonteiroDeFuncao) ();
void setup()
{
  lcd.begin(16, 2);
  Serial.begin(9600);
  Serial.flush(); // Limpar receber buffer.
  pinMode(acaba, INPUT_PULLUP);
  pinMode(desce, INPUT_PULLUP);
  pinMode(sobe, INPUT_PULLUP);
  pinMode(LED_Alerta, OUTPUT);
  pinMode(LED_Direito, OUTPUT);
  pinMode(LED_Esquerdo, OUTPUT);
}

void loop()
{
  PonteiroDeFuncao = StateConect;
  while (1)
  {
    while (config)
    {
      /**
      * chama a função apontada pelo ponteiro de função (logo, chama o estado
      * corrente)
      */
      (* PonteiroDeFuncao) ();
    }
    while (run)
    {
      int state_0 = 0;
      int state_1 = 1;
      digitalWrite(LED_Alerta, LOW);
      while (stim)
      {
        if (digitalRead(sobe) == LOW)
        {
          digitalWrite(sobe, HIGH);
          state_0 = 1;
          lcd.clear();
          lcd.print("Extensao");
          digitalWrite(LED_Direito, HIGH);
        }
        else
          if (digitalRead(desce) == LOW)
          {
            digitalWrite(desce, HIGH);
            state_0 = 2;
            lcd.clear();
            lcd.print("Flexao");
            digitalWrite(LED_Esquerdo, HIGH);
          }
        else
          if (digitalRead(acaba) == LOW)
          {
            digitalWrite(acaba, HIGH);
            state_0 = 3;
          }
        else
        {
          state_0 = 0;
        }
        if (state_1 != state_0)
        {
          Serial.print(state_0);
          state_1 = state_0;
          lcd.clear();
          lcd.print("    Pressione   ");
          lcd.setCursor(0, 1);
          lcd.print("<-Flex     Ext->");
          digitalWrite(LED_Esquerdo, LOW);
          digitalWrite(LED_Direito, LOW);
        }
        if (state_0 == 3)
        {
          Serial.print(state_0);
          stim = false;
          lcd.clear();
          lcd.print("Fim");
        }
        delay(30);
      }
      delay(250);
      digitalWrite(LED_Alerta, HIGH);
      delay(250);
    }
  }
  system("PAUSE");
}

/*-------------------------------------------------------------------------*/
/**
* @brief      Estado inicial de conexão das portas
*/
void StateConect(void)
{
  lcd.clear();
  lcd.setCursor(0, 1);
  lcd.print(" EMA FES-Rowing ");
  delay(1000);
  lcd.setCursor(0, 0);
  lcd.print("   Conectando...");
  /**
  * verificando conexão
  */
  bool conexao = false;
  while (conexao == false)
  {
    if (Serial.available())
    {
      flag = Serial.readString();
      if (flag.equals("a"))
      {
        conexao = true;
        PonteiroDeFuncao = StateSetMov;
      }
    }
  }
}

/**
* @brief      { Estado que seleciona a quantidade de movimento:
*             (extensão)e/ou(flexão) }
*/
void StateSetMov(void)
{
  // para a funcao
  // int mode = 2;
  text = "Sel.Movimento(s):";
  lcd.clear();
  lcd.print(text);
  variavel = mode;
  passo = 1;
  unidade = "";
  print = false;
  mode = func_set_parametro(variavel, passo, text, unidade, print);
  if (mode == 0)
    mode = 3;
  if (mode == 4)
    mode = 1;
  if (mode == 1)
  {
    lcd.setCursor(0, 1);
    lcd.print("Extensao       ");
    channels = B00000011;
  }
  if (mode == 2)
  {
    lcd.setCursor(0, 1);
    lcd.print("         flexao");
    channels = B00001100;
  }
  if (mode == 3)
  {
    lcd.setCursor(0, 1);
    lcd.print("Extensao+flexao");
    channels = B00001111;
  }
  delay(250);
  if (digitalRead(acaba) == LOW)
  {
    if (mode == 2)
      PonteiroDeFuncao = SetAuxCH34;
    else
      PonteiroDeFuncao = SetAuxCH12;
    delay(250);
  }
}

/**
* @brief      { Estado para selecionar se o canal 12 (CH12) necessita de canal
*             auxiliar, se sim: o canal auxíliar será o CH56 }
*/
void SetAuxCH12(void)
{
  text = "Ext.c/ Auxiliar?";
  lcd.clear();
  lcd.print(text);
  passo = 1;
  unidade = "";
  variavel = aux_CH12;
  print = false;
  variavel = func_set_parametro(variavel, passo, text, unidade, print);
  if (variavel == 0)
  {
    aux_CH12 = false;
    lcd.setCursor(0, 1);
    lcd.print("Nao             ");
    set_channels = B00000000;
    delay(250);
  }
  if (variavel == 1)
  {
    aux_CH12 = true;
    lcd.setCursor(0, 1);
    lcd.print("Sim, CH 5/6");
    set_channels = B00110000;
    delay(250);
  }
  if (digitalRead(acaba) == LOW)
  {
    channels = channels | set_channels;
    if (mode == 1)
      PonteiroDeFuncao = StateSetAmp;
    else
      PonteiroDeFuncao = SetAuxCH34;
    delay(250);
  }
}

/**
* @brief      { Estado para selecionar se o canal 12 (CH12) necessita de canal
*             auxiliar, se sim: o canal auxíliar será o CH56 }
*/
void SetAuxCH34(void)
{
  text = "Flex.c/ Auxiliar?";
  lcd.clear();
  lcd.print(text);
  passo = 1;
  unidade = "";
  variavel = aux_CH34;
  print = false;
  variavel = func_set_parametro(variavel, passo, text, unidade, print);
  if (variavel == 0)
  {
    aux_CH34 = false;
    lcd.setCursor(0, 1);
    lcd.print("Nao             ");
    set_channels = B00000000;
    delay(250);
  }
  if (variavel == 1)
  {
    aux_CH34 = true;
    lcd.setCursor(0, 1);
    lcd.print("Sim, CH 7/8");
    set_channels = B11000000;
    delay(250);
  }
  if (digitalRead(acaba) == LOW)
  {
    channels = channels | set_channels;
    PonteiroDeFuncao = StateSetAmp;
  }
  delay(250);
}

/**
* @brief      Estado para selecionar as amplitudes de todos os canais
*             selecionados anteriormente
*/
void StateSetAmp(void)
{
  if ((B00000011 & channels) > 0)
  {
    digitalWrite(acaba, HIGH);
    delay(250);
    while (digitalRead(acaba) == HIGH)
    {
      text = "CorrenteExtensao";
      variavel = corrente_CH12;
      passo = 2;
      unidade = "mA (CH [1,2])";
      print = true;
      corrente_CH12 = func_set_parametro(variavel, passo, text, unidade, print);
      corrente_CH56 = corrente_CH12;
      delay(250);
    }
  }
  if ((B00110000 & channels) > 0)
  {
    digitalWrite(acaba, HIGH);
    delay(250);
    while (digitalRead(acaba) == HIGH)
    {
      text = "Corrente Aux Ext";
      variavel = corrente_CH56;
      passo = 2;
      unidade = "mA (CH [5,6])";
      print = true;
      corrente_CH56 = func_set_parametro(variavel, passo, text, unidade, print);
      delay(250);
    }
  }
  if ((B00001100 & channels) > 0)
  {
    digitalWrite(acaba, HIGH);
    delay(250);
    while (digitalRead(acaba) == HIGH)
    {
      text = "Corrente Flexao";
      variavel = corrente_CH34;
      passo = 2;
      unidade = "mA (CH [3,4])";
      print = true;
      corrente_CH34 = func_set_parametro(variavel, passo, text, unidade, print);
      corrente_CH78 = corrente_CH34;
      delay(250);
    }
  }
  if ((B11000000 & channels) > 0)
  {
    digitalWrite(acaba, HIGH);
    delay(250);
    while (digitalRead(acaba) == HIGH)
    {
      text = "Corrente AuxFlex";
      variavel = corrente_CH78;
      passo = 2;
      unidade = "mA (CH [7,8])";
      print = true;
      corrente_CH78 = func_set_parametro(variavel, passo, text, unidade, print);
      delay(250);
    }
  }
  PonteiroDeFuncao = StateSetLP;
}

/**
* @brief      Estado para selecionar a Largura de Pulso (única para todos os
*             canais)
*/
void StateSetLP(void)
{
  text = "Largura de Pulso:";
  variavel = pw;
  passo = 10;
  unidade = "ms";
  print = true;
  pw = func_set_parametro(variavel, passo, text, unidade, print);
  delay(250);
  if (digitalRead(acaba) == LOW)
  {
    PonteiroDeFuncao = StateSetFreq;
    digitalWrite(acaba, HIGH);
  }
}

/**
* @brief      Estado para selecionar a Frequencia (única para todos os canais)
*/
void StateSetFreq(void)
{
  text = "Frequencia:";
  variavel = freq;
  passo = 5;
  unidade = "Hz";
  print = true;
  freq = func_set_parametro(variavel, passo, text, unidade, print);
  delay(250);
  if (digitalRead(acaba) == LOW)
  {
    PonteiroDeFuncao = Send;
    digitalWrite(acaba, HIGH);
  }
}

/**
* @brief      Estado para enviar os parâmetros para o Raspberry
*/
void Send(void)
{
  // Estado Inicial
  if (B00000000 == channels)
    mode = 0;
  // Extensão
  if (B00000011 == channels)
    mode = 1;
  // Flexão
  if (B00001100 == channels)
    mode = 2;
  // Extensão + Flexão
  if (B00001111 == channels)
    mode = 3;
  // (Extensão & Aux_Ext)
  if (B00110011 == channels)
    mode = 4;
  // (Extensão & Aux_Ext) + Flexão
  if (B00111111 == channels)
    mode = 5;
  // (Flexão & Aux_Flex)
  if (B11001100 == channels)
    mode = 6;
  // Extensao + (Flexão & Aux_Flex)
  if (B11001111 == channels)
    mode = 7;
  // (Extensão & Aux_Ext) + (Flexão & Aux_Flex)
  if (B11111111 == channels)
    mode = 8;
  // enviando dados pela serial
  Serial.print("c"); // marcador de corrente ch12
  func_dim_string_to_Stim(corrente_CH12);
  Serial.print("d"); // marcador de corrente ch34
  func_dim_string_to_Stim(corrente_CH34);
  Serial.print("e"); // marcador de corrente ch56
  func_dim_string_to_Stim(corrente_CH56);
  Serial.print("x"); // marcador de corrente ch78
  func_dim_string_to_Stim(corrente_CH78);
  Serial.print("p"); // marcador de largura de pulso
  func_dim_string_to_Stim(pw);
  Serial.print("f"); // marcador de frequecia
  func_dim_string_to_Stim(freq);
  Serial.print("m"); // marcador do modo
  func_dim_string_to_Stim(mode);
  lcd.clear();
  lcd.print("Enviando");
  delay(2000);
  PonteiroDeFuncao = waiting;
  lcd.setCursor(0, 1);
  lcd.print(channels, BIN);
  lcd.setCursor(15, 1);
  lcd.print(mode);
}

/**
* @brief      Estado para aguardar a confirmação de recebimento pelo Raspberry
*/
void waiting(void)
{
  if (Serial.available())
  {
    flag = Serial.readString();
    if (flag.equals("a"))
    {
      lcd.clear();
      lcd.print("Iniciando estimulacao");
      delay(1000);
      PonteiroDeFuncao = Runing;
    }
  }
}

/**
* @brief      Estado para iniciar o modo Run no loop()
*/
void Runing(void)
{
  run = true;
  config = false;
}

/**
* @brief      Função para acrescentar/diminuir valores da variavel, conforme o
*             passo escolhido, através dos botões do Handle
*
* @param      variavel  Define qual é a varialvel a ser alterada
* @param      passo     Define o passo de acrescimo/decrescimo
* @param      text      Texto informado no display
* @param      unidade   Unidade de medida informada no display
* @param      print     Indica se imprime as mensagens no display (true/false)
*
* @return     Retorna o valor da variavel definida através dos botções
*/
int func_set_parametro(int variavel, int passo, String text, String unidade, bool print)
{
  int BTN_Down = HIGH;
  int BTN_Up = HIGH;
  int lastReadingDown = HIGH;
  int lastReadingUp = HIGH;
  long lastSwitchTimeDown = 0;
  long lastSwitchTimeUp = 0;
  long longSwitchTime = 500;
  long shortSwitchTime = 100;
  long switchTimeDown = 500;
  long switchTimeUp = 500;
  BTN_Up = digitalRead(sobe);
  BTN_Down = digitalRead(desce);
  if (print == true)
  {
    lcd.clear();
    lcd.print(text);
    lcd.setCursor(0, 1);
    lcd.print(variavel);
    lcd.print(" ");
    lcd.print(unidade);
    lcd.print("   ");
    delay(250);
  }
  if (BTN_Up == LOW && ((((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0) || lastReadingUp == HIGH))
  {
    // digitalWrite (sobe, HIGH);
    if (((millis() - lastSwitchTimeUp) > switchTimeUp) && lastSwitchTimeUp != 0)
    {
      switchTimeUp = shortSwitchTime;
    }
    lastSwitchTimeUp = millis();
    lastReadingUp = BTN_Up;
    variavel = variavel + passo;
    delay(250);
  }
  if (BTN_Up == HIGH)
  {
    lastReadingUp = BTN_Up;
    switchTimeUp = longSwitchTime;
    lastSwitchTimeUp = 0;
  }
  if (BTN_Down == LOW &&
  ((((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0) || lastReadingDown == HIGH))
  {
    // digitalWrite (desce, HIGH);
    if (((millis() - lastSwitchTimeDown) > switchTimeDown) && lastSwitchTimeDown != 0)
    {
      switchTimeDown = shortSwitchTime;
    }
    lastSwitchTimeDown = millis();
    lastReadingDown = BTN_Down;
    variavel = variavel - passo;
    if (variavel < 0)
    {
      variavel = 0;
    }
    delay(250);
  }
  if (BTN_Down == HIGH)
  {
    lastReadingDown = BTN_Down;
    switchTimeDown = longSwitchTime;
    lastSwitchTimeDown = 0;
  }
  return variavel;
}

/*-------------------------------------------------------------------------*/
/**
* @brief      Configura o envio de parametros sempre com 3 algarismo
*
* @param      entrada  Valor a ser convertido para 3 algarismo
*
* @return     Valor convertido com 3 algarismo
*/
void func_dim_string_to_Stim(int entrada)
{
  if (qtdAlgarismos(entrada) == 3)
  {
    Serial.print(entrada);
  }
  else
    if (qtdAlgarismos(entrada) == 2)
    {
      Serial.print(0);
      Serial.print(entrada);
    }
  else
    if (qtdAlgarismos(entrada) == 1)
    {
      Serial.print(0);
      Serial.print(0);
      Serial.print(entrada);
    }
}

/*-------------------------------------------------------------------------*/
/**
* @brief      Conta a quantidade de Algarismo
*
* @param      numero  Entrada a ser avaliada
*
* @return     Quatidade de algarismo
*/
int qtdAlgarismos(int numero)
{
  int cont = 0;
  while (numero != 0)
  {
    numero /= 10;
    cont++;
  }
  return cont;
}
