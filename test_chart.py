import flet as ft
from flet.plotly_chart import PlotlyChart
import plotly.graph_objects as go

def main(page: ft.Page):
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])
    page.add(PlotlyChart(fig, expand=True))

if __name__ == "__main__":
    ft.app(target=main)
