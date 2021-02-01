# DialTest

With the tremendous advancement of recurrent neural network (RNN), dialogue systems have achieved significant development. Many RNN-driven dialogue systems, such as Siri, Google Home, and Alexa, have been deployed to assist various tasks. However, accompanying this outstanding performance, RNN-driven dialogue systems, which are essentially a kind of software, could also produce erroneous behaviors and result in massive losses. Meanwhile, the complexity and intractability of RNN models that power the dialogue systems make their testing challenging.



We design and implement DialTest, the first RNNdriven dialogue system testing tool. DialTest employs a series of transformation operators to make realistic changes on seed data while preserving their oracle information properly. To improve the efficiency of detecting faults, DialTest further adopts Gini impurity to guide the test generation process. To validate DialTest, we conduct extensive experiments. We first experiment it on two fundamental tasks, i.e., intent detection and slot filling, of natural language understanding. The experiment results show that DialTest can effectively detect hundreds of erroneous behaviors for different RNN-driven natural language understanding (NLU) module of dialogue systems and improve their accuracy via retraining with the generated data. Further, we conduct a case study on an industrial dialogue system to investigate the performance of DialTest under the real usage scenario. The study shows DialTest can detect errors and improve the robustness of RNN-driven dialogue systems effectively.

## Transformations

- Synonym Replacement (SR): The operators of this family transform the sentence by replacing an individual word with its synonyms, which keeps the meaning of this sentence not changed.
- Back Translation (BT): The operators of this family translate the target sentence into an intermediate language and then translates it back to the original language.
- Word Insertion (WI): The operators of this family transformthe sentence by inserting words with the pre-trained language model.

![截屏2021-02-01 上午9.15.13](/Users/liuzixi/Library/Application Support/typora-user-images/截屏2021-02-01 上午9.15.13.png)



