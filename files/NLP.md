$$BLEU_{n-gram} = BP \times exp(P_n),\ BLEU = BP \times exp(\dfrac{\sum_{i=1}^NP_n}{N}), \\ 
P_n = \dfrac{\sum_{n-gram \in \hat{y}}{Counter_{Clip}(n-gram)}}{\sum_{n-gram \in \hat{y}}{Counter(n-gram)}}, 
BP = \begin{cases}1, & L_{out} > L_{ref} \\ exp(1 - \dfrac{L_{ref}}{L_{out}}), & otherwise \end{cases}$$

$$p_n = \dfrac{\underset{\mathcal{C} \in \{Candidates\}}{\sum}\underset{n-gram \in \mathcal{C}}{\sum} Count_{clip}(n-gram)}{\underset{\mathcal{C‘} \in \{Candidates\}}{\sum}\underset{n-gram‘ \in \mathcal{C’}}{\sum} Count(n-gram’)}, BP = \begin{cases}1, & if\ c > r \\ e^{1 - \frac{r}{c}}, & if\ c \leq r \end{cases} \\ 
BLEU = BP \times exp(\sum_{n=1}^{N}\omega_nlog\ p_n), log\ BLEU = min(1 - \frac{r}{c}, 0) + \sum_{n=1}^N\omega_nlog\ p_n$$

$c = \sum_{i=1}^ML_{out}^i, r = \sum_{i=1}^ML_{ref}^i, \underset{y_i^j, j = [1,...,J]}{arg\ min}{|L_{out}^i - {L_{ref}^i}^j|_1}$

```python
iterables = tee(sequence, n)
for i, sub_iterable in enumerate(iterables): # For each window,
    for _ in range(i):  # iterate through every order of ngrams
        next(sub_iterable, None)  # generate the ngrams within the window
n_gram = zip(*iterables) # Unpack and flattens the iterables
```



