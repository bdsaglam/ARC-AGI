# Conclusion

Generated training data should only be used to assess confidence in the generated solver code. It should not be used to actually develop the solver.

# Strategies

When doing codegen the generalizability of the code is very important to measure, else we will have too high belief in code that solves all test examples and we will fail to see it not generalizing to test cases that may introduce new mechanics.

One way - but not the only way - to solve this is to generate more train input/output pairs.

A key thing to note when generating additional training data (input/output pairs) is that we uniquely DO NOT know the solution at the time we need to generate these cases. If we knew the solution we could easily deploy a wide variety of techniques. But because we do not know, only a few techniques remain:
- Color renaming
- Adding distractors
- Padding
- Reflection
- Rotation

# Assessment

Rotation/reflection fail on many problems. For example https://arcprize.org/play?task=31f7f899 which assumes a right-to-left sorting. A reflection of 180-degree rotation would brake this problem. Furthermore, a 90-degree rotation would introduce unnecessary complexity. Similarly on https://arcprize.org/play?task=7b5033c1 you may first thing that rotations would clear up simplistic assumption of top-to-bottom and distill them into the stricter "end-to-end", but in deriving the orientation of the output (vertical or horizontal) a 90-degree rotation on training input 2 actually makes this problem unsolvable (you can't determing output orientation).

Rotation can however be very helpful, like in https://arcprize.org/play?task=16de56c4 as an example. And, both rotation and reflection in this case works. In my tests with gpt-5.2-medium this problem is unsolvable, despite ~half of the time the solver gets all the training examples right. It fails to generalize to the right-to-left case - which would have been caught through generated training data.

Even color renaming, which seems harmless at first sight, brakes some problems. An example is https://arcprize.org/play?task=332f06d7 where renaming red and black makes the problem unsolvable.

I believe there are no completely safe ways to generate training data without knowing the solution transform. That said, generating training data can be helpful in building *some* confidence in the solver. If it also solves generated training data it is more likely to generalize, but even if failing the generated training data it still *can* be right.



