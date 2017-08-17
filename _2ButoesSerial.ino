
// constants won't change. They're used here to set pin numbers:
const int buttonPin1 = 7;
const int buttonPin2 = 8;// the number of the pushbutton pin
const int ledPin =  13;      // the number of the LED pin

// variables will change:
int buttonState1 = 0;         // variable for reading the pushbutton status
int buttonState2 = 0;
int state; // pode ser 0,1 ou 2 a depender do boto apertado

void setup() {

  Serial.begin(9600);

  pinMode(ledPin, OUTPUT);      
  pinMode(buttonPin1, INPUT);     
  pinMode(buttonPin2, INPUT);     

}

void loop(){
  // read the state of the pushbutton value:
  buttonState1 = digitalRead(buttonPin1);
  buttonState2 = digitalRead(buttonPin2);
  // check if the pushbutton is pressed.
  // if it is, the buttonState is HIGH:
  if(buttonState2 == HIGH && buttonState1 == HIGH){
    state = 3;
  }
  else if (buttonState1 == HIGH) {     
    state = 1;
  } 
  else if (buttonState2 == HIGH) {     
    state = 2;
  }

  else{
    state = 0;
  }

  Serial.println(state);
  delay(50);
}

