from bokeh.plotting import figure, show
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CustomJS, Slider, Range1d
from bokeh.layouts import column, row
from bokeh.palettes import *
from bokeh.transform import factor_cmap


fruits = ['Apples', 'Bananas', 'Kiwis', 'Pears', 'Watermelons']
counts = [10, 5, 24, 12, 4]
source = ColumnDataSource(data=dict(x=fruits, top=counts))
color_map = factor_cmap(
    field_name='x', palette=cividis(len(fruits)), factors=fruits)

plot = figure(width=400, height=400, x_range=fruits)
plot.y_range.start = 0

plot.vbar(source=source, width=0.8, bottom=0, color=color_map)

slider = Slider(start=1, end=100, value=1, step=1, title="Bananas Count")

callback = CustomJS(args=dict(source=source), code="""
  const x = source.data.x
  const y = source.data.top

  var new_y = []
  for (var i = 0; i < y.length; i++) {
    if(i == 1){
      new_y.push(this.value)
    }
    else{
      new_y.push(y[i])
    }
  }

  source.data = {'x':x, 'top':new_y}

  console.log(source.data.top)
""")

slider.js_on_change('value', callback)

layout = column([slider, plot])
show(layout)
