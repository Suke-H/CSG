from graphviz import Digraph

G = Digraph(format="png")
G.attr("node", shape="square", style="filled")
G.edge("start","state1",label="0.8")
G.edge("start","state2",label="0.2")
G.edge("state1","state1",label="0.5")
G.edge("state2","state2", label="0.8")
G.edge("state1","state2",label="0.5")
G.edge("state2","end",label="0.2")
G.edge("end","count",label="1.0")
G.edge("count","start",label="1.0")
G.node("start", shape="circle", color="pink")
G.render("test")