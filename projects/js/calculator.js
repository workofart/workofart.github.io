var num1 = 0;
var num2 = 0;
var operator;
var displayNum = document.getElementById('display');
var decimal =0;

/*********************** Operation Functions**************/
function addition () {
	var sum;
	sum = num1 + num2;
	return sum;
}

function multiply () {
	var product;
	product = num1 * num2;
	return product;
}

function divide () {
	var quotient;
	quotient = num2 / num1; // switched order due to accomodate storage problems 
	return quotient;
}

function subtract () {
	var difference;
	difference = num2 - num1; // switched order due to accomodate storage problems 
	return difference;
}

function inverse () {
	displayNum.innerHTML = -Number(displayNum.innerHTML);
	this.num1 = Number(displayNum.innerHTML); // update num1 with the new display num
}

function percentage () {
	displayNum.innerHTML = Number(displayNum.innerHTML) * 0.01;
	this.num1 = Number(displayNum.innerHTML); // update num1 with the new display num
}

/********************* Other Functions *************************/

// Append the current num at the end of the num1
function display (num) {
	// this.num1 = Number(displayNum.innerHTML);
	if (num === ".") {
		decimal++; // trigger decimal, increment counter
	}
	else if (decimal !== 0){		
	// decrease the num by decimal times
		for (var i = 0; i < decimal;i++){
			num = num * 0.1;
		}
		this.num1  = this.num1 + num;
		decimal++;
	}
	else {// else add the num to num1
		this.num1 = this.num1 * 10 + num;
	}
	displayNum.innerHTML = this.num1;
}

// assign operators
// everytime any operator has been pressed, store the display num to num1 variables for future use
function assignOperator(operator){
	this.operator = operator;
	// asssign the current display value to num1
	// this.num1 = Number(displayNum.innerHTML);
	this.num2 = this.num1;
	// prepare for the next input
	decimal = 0; // reset decimal
	this.num1 = 0;

}

// Evaluate the expression based on the current operator
function evaluateExp () {
	if (this.operator === "+")
		displayNum.innerHTML = addition();
	else if (this.operator === "-")
		displayNum.innerHTML = subtract();
	else if (this.operator ==="*")
		displayNum.innerHTML = multiply();
	else if (this.operator ==="/")
		displayNum.innerHTML = divide();
	else
		displayNum.innerHTML = num2;

	// store the current display num to num1 to prepare for the next iteration
	this.num1 = Number(displayNum.innerHTML);
}

// Clear the two number variables to start over
function reset () {
	this.num1 = 0;
	this.num2 = 0;
	this.operator = "";
	displayNum.innerHTML = 0;
	decimal = 0;
}