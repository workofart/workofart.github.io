
// getElementById Simplified
function getId (id) {
	return document.getElementById(id);
}

// get tag  simplified
function getTag (tag) {
	return document.getElementsByTagName(tag);
}

function clearCanvas() {
	var canvas = getId("myCanvas");
	var ctx= canvas.getContext("2d");
	ctx.clearRect(0,0,900000,900000);
}

function untest1() {
	var button = getId("test1");
	button.innerHTML = "test1";
	canvas.innerHTML = "";
}

function zoomIn () {
	var canvas = getId("myCanvas");
	var ctx = canvas.getContext("2d");
	// var grd = ctx.createLinearGradient(0,40,0,0);
	// grd.addColorStop(0,"black");
	// grd.addColorStop(1,"white");
	// ctx.fillStyle=grd;
	ctx.strokeRect(20,20,150,100);
	ctx.scale(1.05,1.05);
	ctx.strokeRect(20,20,150,100);
}

function zoomOut () {
	var canvas = getId("myCanvas");
	var ctx = canvas.getContext("2d");
	// var grd = ctx.createLinearGradient(0,40,0,0);
	// grd.addColorStop(0,"black");
	// grd.addColorStop(1,"white");
	// ctx.fillStyle=grd;
	ctx.strokeRect(20,20,150,100);
	ctx.scale(0.95,0.95);
	ctx.strokeRect(20,20,150,100);
}

var i = 1;                     //  set your counter to 1
function animatedZoom(){
(function myLoop (i) {     
   setTimeout(function () {  
      zoomOut();
      //  your code here                
      if (--i) myLoop(i);      //  decrement i and call myLoop again if i > 0
   }, 100)
})(40);
}

function addShadow () {
	var canvas = getId("myCanvas");
	var ctx = canvas.getContext("2d");
	ctx.shadowBlur= 10;
	ctx.shadowColor = "grey";
}
	
