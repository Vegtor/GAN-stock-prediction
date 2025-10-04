import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def trading_points(data, coef=0.05):
    temp = data[0]
    down_points = list()
    up_points = list()
    bought = 0
    j = 1

    for i in range(1, data.shape[0]):
        if (bought == 1 and temp > data[i]) or (bought == 1 and i == data.shape[0] - 1):
            up_points.append(i - 1)
            bought = 0
        if bought == 0 and temp < data[i]:
            if i == data.shape[0] - 1:
                break
            down_points.append(i - 1)
            bought = 1
        temp = data[i]

    while True:
        if j >= len(down_points):
            break
        change = (data[up_points[j]] - data[down_points[j]]) / data[down_points[j]]
        if change <= coef:
            if data[up_points[j]] < data[up_points[j - 1]]:
                up_points.pop(j)
                down_points.pop(j)
            elif j != len(down_points) - 1 and data[up_points[j + 1]] > data[up_points[j]]:
                up_points.pop(j)
                down_points.pop(j + 1)
            else:
                up_points.pop(j - 1)
                down_points.pop(j)
        else:
            j = j + 1

    return down_points, up_points

def setup_figure(data_list, names, axis, colours, widths, title, font_size, dates=None, plot_w = 500, plot_h = 500, legend_x = "left", legend_y = "top", pos_x = 1, pos_y = 1):
    n = len(data_list[0])
    fig1 = go.Figure()
    if dates is None:
        for i in range(len(data_list)):
            fig1.add_trace(go.Scatter(x=list(range(n)), y=data_list[i], name=names[i],
                                      line=dict(color=colours[i], width=widths[i])))
    else:
        for i in range(len(data_list)):
            fig1.add_trace(go.Scatter(x=dates, y=data_list[i], name=names[i],
                                      line=dict(color=colours[i], width=widths[i])))

    fig1.update_layout(
        title=title,
        width=plot_w,
        height=plot_h,
        xaxis=dict(
            title=axis[0],
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            title=axis[1],
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        font=dict(family="Helvetica, Arial, sans-serif", size=font_size, color="black"),
        showlegend=True,
        template='plotly_white',
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            yanchor=legend_y,
            xanchor=legend_x,
            x=pos_x,
            y=pos_y
        ),
        plot_bgcolor='white',
    )
    return fig1

def create_graph(data_list, names, axis, colours, widths, title, font_size, dates=None, plot_w = 500, plot_h = 500, legend_x = "left", legend_y = "top", pos_x = 1, pos_y = 1):
    fig1 = setup_figure(data_list, names, axis, colours, widths, title, font_size, dates, plot_w, plot_h, legend_x, legend_y, pos_x, pos_y)
    fig1.show()

def graph_with_points(figure, points_list, y_data, dates, names, colours):
    fig1 = figure
    for i in range(len(points_list)):
        fig1.add_trace(go.Scatter(x=[dates[j] for j in points_list[i]],
                                  y=[y_data[j] for j in points_list[i]],
                                  name=names[i],
                                  mode='markers',
                                  marker=dict(
                                      symbol='square',
                                      size=10,
                                      color='rgba(0,0,0,0)',
                                      line=dict(
                                          color=colours[i],
                                          width=2
                                                )
                                            ),
                                  )
                       )
    fig1.show()