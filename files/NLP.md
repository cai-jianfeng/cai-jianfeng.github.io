$$BLEU_{n-gram} = BP \times exp(P_n),\ BLEU = BP \times exp(\dfrac{\sum_{i=1}^NP_n}{N}), \\ 
P_n = \dfrac{\sum_{n-gram \in \hat{y}}{Counter_{Clip}(n-gram)}}{\sum_{n-gram \in \hat{y}}{Counter(n-gram)}}, 
BP = \begin{cases}1, & L_{out} > L_{ref} \\ exp(1 - \dfrac{L_{ref}}{L_{out}}), & otherwise \end{cases}$$

