# Training Speed of SGD, Momentum, Nesterov SGD, RMSprop, and Adam

This work is inspired by [Alec Radford](https://twitter.com/AlecRad/media). His first [work](https://www.reddit.com/r/MachineLearning/comments/2gopfa/visualizing_gradient_optimization_techniques/cklhott/) was posted on Reddits on 2014. However, there is no Adam algorithm which was published on 2015. As a result, I I implemented this animation. 

Animation effects are referenced from
  * [Louis Tiao's work](http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/)
### My reproduced work
![N|Solid](https://github.com/Brandon-HY-Lin/neural_networks_for_machine_learning/blob/master/week_06/momentum_methods/saddle_animation_3d.gif?raw=true)

### Alec Radford's work
![N|Solid](https://github.com/Brandon-HY-Lin/neural_networks_for_machine_learning/blob/master/week_06/momentum_methods/alec_radford_animation.gif?raw=true)

### Execution
```sh
$ python saddle_class.py
```

### Saving GIF Animation
Simply set variable IS_SAVE to True in the python file
```python
IS_SAVE=True
```

### License

Apache License, Version 2.0
