import plotly.graph_objects as go
try:
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])
    img_bytes = fig.to_image(format="png")
    print("Kaleido export successful")
except Exception as e:
    print(f"Kaleido export failed: {e}")
