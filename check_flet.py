import flet
try:
    from flet.plotly_chart import PlotlyChart
    print("Found in flet.plotly_chart")
except ImportError:
    print("Not in flet.plotly_chart")

try:
    from flet import PlotlyChart
    print("Found in flet")
except ImportError:
    print("Not in flet")

try:
    import flet.plotly
    print("Found flet.plotly module")
except ImportError:
    print("No flet.plotly module")
