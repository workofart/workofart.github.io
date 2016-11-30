/************** Bar Graph ********************/

var wins = [
[ [2006,13],[2007,11],[2008,15],[2009,15],
[2010,18],[2011,21],[2012,28]
]
];
var wins2 = [[[0,28]],[[1,28]],[[2,21]],[[3,20]],[[4,19]]];

var years = [
    [0, "2006"],
    [1, "2007"],
    [2, "2008"],
    [3, "2009"],
    [4, "2010"],
    [5, "2011"],
    [6, "2012"]
];

var teams = [
    [0, "MCI"],
    [1, "MUN"],
    [2, "ARS"],
    [3, "TOT"],
    [4, "NEW"]
];

function barChart () {
Flotr.draw(document.getElementById("chart"), wins2, {
    title: "Premier League Wins (2011-2012)",
    colors: ["#89AFD2", "#1D1D1D", "#DF021D", "#0E204B", "#E67840"],
    bars: {
        show: true,
        barWidth: 0.5,
        shadowSize: 0,
        fillOpacity: 1,
        lineWidth: 0
    },
    yaxis: {
        min: 0,
        tickDecimals: 0
    },
    xaxis: {
        ticks: teams
    },
    grid: {
        horizontalLines: false,
        verticalLines: false
    }
});};


/************** Line Graph ********************/

var zero = [];
for (var yr=1959; yr<1962; yr++) { zero.push([yr, 0]); };

var co2 = [
    [ 1959, 315.97 ],
    [ 1960, 316.91 ],
    [ 1961, 317.64 ],
    [ 1962, 318.45 ]];

var temp = [
    [ 1959,  0.0776 ],
    [ 1960,  0.0280 ],
    [ 1961,  0.1028 ],
    [ 1962,  0.1289 ]];

function lineChart() {
	Flotr.draw(
    document.getElementById("chart"),
    [
        { data: zero,label: "20<sup>th</sup> Century Baseline Temperature", lines: {show:true, lineWidth: 1}, yaxis: 2,
          shadowSize: 0, color: "#545454" },
        { data: co2, label: "CO<sub>2</sub> Concentration (ppm)",lines: {show:true} },
        { data: temp, label: "Yearly Temperature Difference (°C)",lines: {show:true}, yaxis: 2 }
    ],{
        title: "Global Temperature and CO2 Concentration (NOAA Data)",
        grid: {horizontalLines: false, verticalLines: false},
        yaxis: {min: 300, max: 400},
        y2axis: {min: -0.15, max: 0.69, tickFormatter: function(val) {return val+" °C";}}
    }
);}