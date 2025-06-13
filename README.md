# CS-6140-Homework-1-solution

Download Here: [CS 6140 Homework 1 solution](https://jarviscodinghub.com/assignment/cs-6140-homework-1-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

1) Probability and Random Variables: State true or false. If true, prove it. If false, either
prove or demonstrate by a counter example. Here Ω denotes the sample space and Ac denotes the
complement of the event A.
1. For any A, B ⊆ Ω such that P(A) > 0, P(Bc
) > 0, P(B|A) + P(A|Bc
) = 1.
2. For any A, B ⊆ Ω such that 0 < P(B) < 1, P(A|B) + P(A|Bc
) = 1.
3. For any A, B ⊆ Ω, P(Bc ∪ (A ∩ B)) + P(Ac ∩ B) = 1.
4. Let {Ai}
n
i=1 be mutually independent. Then, P(∪
n
i=1Ai) = Pn
i=1 P(Ai).
5. Let {(Ai
, Bi)}
n
i=1 be mutually independent, i.e., (Ai
, Bi) is independent from (Aj
, Bj ) for
every i 6= j. Then P(A1, . . . , An|B1, . . . , Bn) = Qn
i=1 P(Ai
|Bi)
2) Discrete and Continuous Distributions: Write down the formula of the probability density/mass functions of random variable X.
1. k-dimensional Gaussian distribution (multi-variate Gaussian), X ∼ N (x; µ, Σ).
2. Laplace distribution with mean µ and variance 2σ
2
.
3. Bernoulli distribution, X ∼ Bernoulli(p), 0 < p < 1.
4. Multinomial distribution with N trials and L outcomes with probabilities θ1, . . . , θL.
5. Dirichlet distribution of order L with parameters α1, . . . , αL.
6. Uniform distribution, X ∼ Unif(a, b), a < b.
7. Exponential distribution, X ∼ Exp(λ), λ > 0.
8. Poisson distribution, X ∼ Poisson(λ), λ > 0.
3) Positive-Definite Matrices: A symmetric matrix A ∈ R
m×m is positive-semidefinite if x
>Ax ≥
0 for every x ∈ R
m, where x 6= 0. Equivalently, A is positive-semidefinite if all eigenvalues of
A are non-negative. Prove or disprove (by a counter example) that the following matrices are
positive-semidefinite.
1. A = B>B for an arbitrary B ∈ R
m×n
2. A =


8 −5 −3
−5 5 0
−3 0 3


1
3. A = B + B> + B>B for an arbitrary B ∈ R
n×n
4) Convexity of Linear Regression: In the class, we studied several models for linear regression.
Let X ∈ R
N×d
and Y ∈ R
N denote matrices of input features and outputs/responses, respectively.
Let θ ∈ R
d denote the vector of unknown parameters.
a) Show that the following objective functions for linear regression are convex with respect to θ.
1. Vanilla/Basic regression: J1(θ) = kY − Xθk
2
2
2. Ridge regression: J2(θ) = kY − Xθk
2
2 + λkθk
2
2
3. Lasso regression: J3(θ) = kY − Xθk
2
2 + λkθk1
b) What conditions do we need to impose on X and/or Y in each of the above cases, so that the
solution for θ be unique?
5) Regression using Huber Loss: In the class, we defined the Huber loss as
`δ(e) = (
1
2
e
2
if |e| ≤ δ
δ|e| − 1
2
δ
2
if |e| > δ
Consider the robust regression model
min
θ
X
N
i=1
`δ(yi − θ
>xi),
where xi and yi denote the i-th input sample and output/response, respectively and θ is the unknown parameter vector.
a) Provide the steps of the batch gradient descent in order to obtain the solution for θ.
b) Provide the steps of the stochastic gradient descent using mini-batches of size 1, i.e., one sample
in each mini-batch, in order to obtain the solution for θ.
6) PAC Confidence Bounds: In the class, we studied the problem of maximum likelihood estimation of a Bernoulli random variable (taking values in {0, 1}), where the true probability of being 1
is assumed to be θ
o
. We showed that using the maximum likelihood estimation on a dataset with
N samples, the ML estimate is given by ˆθ =
PN
i=1 xi/N. Moreover, we showed that
P

|
ˆθ − θ
o
| ≥ 

≤ 2e
−N2
.
Consider the example of flipping a coin N times with the true probability of ‘Head’ to be θ
o
. How
many trials (flipping the coin) we need to have in order to be confident that with probability at
least 0.95, the estimate of the maximum likelihood for the probability of ‘Head’ will be within 0.1
distance of the true value?
7) Probabilistic Regression with Prior on Parameters: Consider the probabilistic model of regression, where p(y|x, θ) is a Normal distribution with mean θ
>x and variance σ
2
, i.e., N (θ
>x, σ2
).
2
We would like to determine θ using a dataset of N samples {(x1, y1), . . . ,(xN , yN )}. Assume we
have prior information about the distribution of θ and our goal is to determine the Maximum A Posteriori (MAP) estimate of θ using the dataset and the prior information. For each of the following
cases, provide the optimization from which we can obtain the MAP solution.
1. θ ∼ N (0, 1/λI), where I denotes the identity matrix.
2. Each element of θ has a Laplace distribution with mean 0 and variance 2/λ2
.
8) MAP estimation for the Bernoulli with non-conjugate priors: Consider a Bernoulli random
variable x with p(x = 1) = θ. In the class, we discussed MAP estimation of the Bernoulli rate
parameter θ with the prior p(θ) = Beta(θ|α, β). We know that, with this prior, the MAP estimate
is given by:
ˆθ =
N1 + α − 1
N + α + β − 2
where N1 is the number of trails where xi = 1 (e.g., heads in flipping a coin), N0 is the number of
trials where xi = 0 (e.g., tails in flipping a coin) and N = N0 + N1 is the total number of trials.
1. Now consider the following prior, that believes the coin is fair, or is slightly biased towards
heads:
p(θ) =



0.5 if θ = 0.5
0.5 if θ = 0.6
0 otherwise
Derive the MAP estimate under this prior as a function of N1 and N.
2. Suppose the true parameters is θ = 0.61. Which prior leads to a better estimate when N is
small? Which prior leads to a better estimate when N is large?
9) Gaussian Naive Bayes: The multivariate normal distribution in k-dimensions, also called the
multi-variate Gaussian distribution and denoted by N (µ, Σ), is parameterized by a mean vector
µ ∈ R
k
and a covariance matrix Σ ∈ R
k×k
, where Σ ≥ 0 is positive semi-definite.
Consisder a classification problem in which the input feature x ∈ R
k
are continuous-valued random variables, we can then use the Gaussian Naive Bayes (GNB) model, which models p(x|y)
using a multivariate normal distribution. The model is given by
y ∼ Bernoulli(φ)
x|y = 0 ∼ N (µ0
, Σ)
x|y = 1 ∼ N (µ1
, Σ)
where we assume that Σ is a diagonal matrix, Σ = diag(σ
2
1
, . . . , σ2
k
). Given a training dataset
{(x
1
, y1
), . . . ,(x
N , yN )}, write down the likelihood (log-likelihood) and derive MLE estimates
for the means µ0
, µ1
, covariance Σ and the class prior φ of the GNB.
10) Linear Regression Implementation:
a) Write down a code in Python whose input is a training dataset {(x
1
, y1
), . . . ,(x
N , yN )} and its
output is the weight vector θ in the linear regression model y = θ
>φ(x), for a given nonlinear
3
mapping φ(·). Implement two cases: i) using the closed-form solution, ii) using a stochastic gradient descent on mini-batches of size m.
b) Consider n-degree polynomials, φ(·) =
1 x x2
· · · x
n

. Download the dataset on the
course webpage and work with ‘dataset1’. Run the code on the training data to compute θ for
n ∈ {2, 3, 5}. Evaluate the regression error on both training and the test data. Report θ, training
error and test error for both implementation (closed-form vs gradient descent). What is the effect
of the size of the mini-batch on the speed and testing error of the solution.
c) Download the dataset on the course webpage and work with ‘dataset2’. Write a code in Python
that applies Ridge regression to the dataset to compute θ for a given λ. Implement two cases:
using a closed-form solution and using a stochastic gradient descent method with mini-batches of
size m. Use K-fold cross validation on the training dataset to obtain the best regularization λ and
apply the optimal θ to compute the regression error on test samples. Report the optimal λ, θ, test
and training set errors for K ∈ {2, 10, N}, where N is the number of samples. In all cases try
n ∈ {2, 3, 5}. How does the test error change as a function of λ and n?

