// 1D Plotting
function relabel1DBursts(data, burstPlot, dfIdx, burstTraces, traceId) {
  dfData = fetchData(dfIdx);
  var curveNumber = data.points[0].curveNumber;
  if (curveNumber == traceId) {
      var targetTrace = burstTraces[data.points[0].pointNumber];
      var color = burstPlot.data[targetTrace].line.color;
  } else if (burstTraces.includes(curveNumber)) {
      var targetTrace = curveNumber;
      var color = data.points[0].data.line.color;
  } else {
      return;
  }
  if (color == 'black') {
      var color_inv = 'red';
      var isBurst = 'True';
      var name = 'Burst';
  } else {
      var color_inv = 'black';
      var isBurst = 'False';
      var name = 'Signal';
  }
  var update = {'name': name, 'line':{color: color_inv}};
  Plotly.restyle(burstPlot, update, [targetTrace]);
}

// 2D and 3D Plotting
function recolorBursts(plotID){
  // Change colors of signal/burst
  var graph = document.getElementById(plotID);
  graph.on('plotly_click', function(data){
    if (data.points[0].data.name == 'Burst') {
      var update = {'name': 'Signal', 'line': {color: 'black'}};
    } else {
      var update = {'name': 'Burst', 'line': {color: 'red'}};
    };
    Plotly.restyle(graph, update, [data.points[0].curveNumber]);
  });
}

function rewriteBursts(divIds){
  // Determine the is_burst column of the current plot(s)
  var isBurst = [];
  var isBurstIdx = 0;
  for (idx=0; idx < divIds.length; idx++) {
    var divId = divIds[idx];
    var graph = document.getElementById(divId);
    for (traceIdx=0; traceIdx < graph.data.length; traceIdx++) {
      if (graph.data[traceIdx].name == 'Burst') {
        isBurst[isBurstIdx] = 'True';
      } else if (graph.data[traceIdx].name == 'Signal') {
        isBurst[isBurstIdx] = 'False';
      };
      isBurstIdx++;
    }
  }
  // Append columns and save out csv
  dfData = fetchData("None");
  dfNew = [];
  header = dfData[0][0]
  dfNew.push(header)
  for (i=0; i < dfData.length; i++){
    for (j=1; j < dfData[i].length; j++){
      dfNew.push(dfData[i][j])
    }
  };
  if ( ! dfNew.includes('is_burst_new')) {
    dfNew[0][dfNew[0].length-1] = 'is_burst_orig';
    dfNew[0].push('is_burst_new');
  }
  for (i=0; i < isBurst.length; i++) {
    dfNew[i+1][dfNew[i+1].length] = isBurst[i];
  }
  saveCsv(dfNew);
}

function saveCsv(dfData){
  // Save out csv file
  var csvRows = [];
  for(var i=0, l=dfData.length; i<l; ++i){
      csvRows.push(dfData[i].join(','));
  }
  var csvString = csvRows.join("\n");
  var a         = document.createElement('a');
  a.href        = 'data:attachment/csv,' + encodeURIComponent(csvString);
  a.target      = '_blank';
  a.download    = 'results.csv';
  document.body.appendChild(a);
  a.click();
}

