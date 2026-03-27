#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int rand_move(int k){

srand(time(NULL));

int r = rand() % (3-1) + 1;

return r;

}


int check_win(int l){
//0 for lose
//1 for win
//2 for draw

if(l==1 && rand_move(l)==2){
return 0;
}

else if(l==1 && rand_move(l) ==3){
return 1;
}

else if(l==2 && rand_move(l) ==3){
return 0;
}

else if(l==2 && rand_move(l) ==1){
return 1;
}

else if(l==3 && rand_move(l) ==1){
return 0;
}

else if(l==3 && rand_move(l) ==2){
return 1;
}

else if(l == rand_move(l)){
return 2;
}

}

int main(){

char Rock[] = u8"🪨";
char Paper[] = u8"📄";
char Scissors[] = u8"✂️";

printf("Rock Paper Scissors\n\n\n");

//1 Rock
//2 Paper
//3 Scissors

int move;

printf("Enter 1 for Rock %s, 2 for Paper %s and 3 for Scissors %s  :", Rock, Paper, Scissors);

scanf("%d", &move);

if(move != 1 && move != 2 && move != 3){
printf("Please Enter a Valid move.\n");
printf("Enter move:");
scanf("%d", &move);
}

switch (move){

	case 1: 
		printf("\nYour Move %s", Rock);
		break;
	case 2: 
		printf("\nYour Move %s", Paper);
		break;
	case 3: 
		printf("\nYour Move %s", Scissors);
		break;
}


switch (rand_move(move)){

	case 1: 
		printf("\nComputer's Move %s", Rock);
		break;
	case 2: 
		printf("\nComputer's Move %s", Paper);
		break;
	case 3: 
		printf("\nComputer's Move %s", Scissors);
		break;
}


if(check_win(move)==1){
printf("\n\nCongratulations!! You won the game.\n");

}

else if(check_win(move)==0){
printf("\n\nYou Lost the game.\n");
}

else if(check_win(move)==2){
printf("\n\nIt's a Draw!\n");
}

printf("Thank you for playing.\n");


return 0;
}
