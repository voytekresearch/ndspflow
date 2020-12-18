// 1D Plotting

function relabel1DBursts(data, burstPlot, plotData, burstTraces, traceId) {
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
  } else {
      var color_inv = 'black';
      var isBurst = 'False';
  }
  var update = {'line':{color: color_inv}};
  Plotly.restyle(burstPlot, update, [targetTrace]);
  cyc = targetTrace-burstTraces[0];
  if ( ! plotData[0].includes('is_burst_new')){
      plotData[0][plotData[0].length-1] = 'is_burst_orig';
      plotData[0].push('is_burst_new');
      for (i=1; i<plotData.length; i++){
          plotData[i][plotData[0].length-1] = plotData[i][plotData[0].length-2]
      }
  }
  plotData[cyc+1][plotData[cyc].length-1] = isBurst;
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

function rewriteBursts(dfData, divIds){
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
  if ( ! dfData.includes('is_burst_new')) {
    dfData[0][dfData[0].length-1] = 'is_burst_orig';
    dfData[0].push('is_burst_new');
  }
  for (i=0; i < isBurst.length; i++) {
    dfData[i+1][dfData[i+1].length] = isBurst[i];
  }
  saveCsv(dfData);
}

function saveCsv(plotData){
  // Save out csv file
  var csvRows = [];
  for(var i=0, l=plotData.length; i<l; ++i){
      csvRows.push(plotData[i].join(','));
  }
  var csvString = csvRows.join("\n");
  var a         = document.createElement('a');
  a.href        = 'data:attachment/csv,' + encodeURIComponent(csvString);
  a.target      = '_blank';
  a.download    = 'results.csv';
  document.body.appendChild(a);
  a.click();
}

