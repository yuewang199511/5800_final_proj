# instruction

This agent can be run in by the following command
`python3 gui_main.py --red-subprocess "my_agent.py" --blue-subprocess "my_agent.py"`

This agent is composed by several different heuristics and consolidated in one evalator

# feature
```python
    evaluator = (
        HeuristicEvaluator(temperature=0.1)
        .register(MyShortestPath(),  weight=1.0) # 2 pass of BFS
        .register(OppShortestPath(), weight=2.0) # 2 pass of BFS
        .register(CenterBias(),      weight=0.1)
    )
```

You can change `weight=0` the remove the effect of one algorithm and increase the weight for increasing its effect.
Higher temperature means the resulted step will be more random rather than following the heuristic result.

The algorithm is complete right now.
