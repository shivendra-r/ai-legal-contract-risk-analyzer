// static/script.js
function renderRiskChart(score) {
  // score = 0..100
  const ctx = document.getElementById('riskChart');
  if (!ctx) return;

  const data = {
    labels: ['Risk', 'Safe'],
    datasets: [{
      data: [score, Math.max(0, 100 - score)],
      backgroundColor: [ '#ef4444', '#e6eefc' ],
      hoverOffset: 6,
      borderWidth: 0
    }]
  };

  window.riskChart = new Chart(ctx, {
    type: 'doughnut',
    data: data,
    options: {
      cutout: '70%',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: {
          label: function(context) {
            return context.label + ': ' + context.parsed + '%';
          }
        }}
      }
    }
  });

  // show center text by drawing on top
  const plugin = {
    id: 'centerText',
    beforeDraw: function(chart) {
      const w = chart.width, h = chart.height, ctx = chart.ctx;
      ctx.restore();
      const fontSize = (h / 7).toFixed(2);
      ctx.font = fontSize + "px Arial";
      ctx.fillStyle = "#111827";
      ctx.textBaseline = "middle";
      const text = score + "%";
      const textX = Math.round((w - ctx.measureText(text).width) / 2);
      const textY = h / 2;
      ctx.fillText(text, textX, textY);
      ctx.save();
    }
  };

  // register plugin once
  if (!Chart.registry.plugins.get('centerText')) {
    Chart.register(plugin);
  }
}

// expose for inline call in result template
window.renderRiskChart = renderRiskChart;
