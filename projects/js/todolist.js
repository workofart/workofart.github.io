function addItem() {
	var content = document.getElementById("itemInput");
	var numElems = numItems() + 1; // adjust discrepancies
	var s = '<td id="item' + numElems + '"><button type="button" class="btn" onclick="deleteItem(' + numElems + ');numItems()"><span class="glyphicon glyphicon-ok"></span></button></td><td></td>';
	var newNode = document.createElement("tr");
	newNode.innerHTML = s;
	// append node
	document.getElementById("itemList").appendChild(newNode);
	var lastC = document.getElementById('itemList').lastChild.lastChild;
	// add the item in the last child node
	lastC.innerHTML = content.value;
	content.value = "";

}

function deleteAllItems() {
	while (numItems() >0){
		document.getElementById('itemList').removeChild(document.getElementById('itemList').lastChild);
	}
	
}
function deleteItem (n1) {
	var str = "item".concat(n1);
	var target = document.getElementById(str).parentNode;
	target.parentNode.removeChild(target);
}

function numItems () {
	var w = document.getElementById('itemList');
	var count = 0; // this will contain the total elements.
	for (var i = 0; i < w.childNodes.length; i++) {
		var node = w.childNodes[i];
		if (node.nodeName == "TR") {
			count++;
		}
	}
	document.getElementById("itemCount").innerHTML = count + " items";
	return count;
}

function exportResults () {
	var A = [['To-do'+ ' List']];  // initialize array of rows with header row as 1st item
	for (var i = 1; i <= numItems(); i++) {
	// var node = w.childNodes[i];
	A.push(["",document.getElementById("item".concat(i)).nextSibling.innerHTML]);
}
var csvRows = [];
for(var j=0,l=A.length; j<l; ++j){
		csvRows.push(A[j].join(',\n'));   // unquoted CSV row
	}
	
	// joining all the rows
	var csvString = csvRows.join("\n\n");
	var a = document.createElement('a');
	a.href     = 'data:attachment/csv;charset=utf8,' + csvString;
	a.target   = '_blank';
	a.download = 'toDoList.csv';
	document.body.appendChild(a);
	a.click();
}

/****************** Testing ***********************/

function test() {

}

function unTest() {
	document.getElementById("item").innerHTML = "Row 1";
}

function testing () {
	var x = document.getElementById("itemInput");
	var text = "";
	var i;
	for (i = 0; i < x.length; i++) {
		text += x.elements[i].value + "<br>";
	}
	document.getElementById("demo").innerHTML = text;
}

