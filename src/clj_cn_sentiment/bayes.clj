(ns clj-cn-sentiment.bayes
  (require [clojure.string :as s]))


(defn count->probability
  "Take an hash map contain the number of occurance in each class,
return the posterior probabilities. The priori probabilities are 
equal for each class."
  [h]
  {:post [(= (count %) (count h))]}
  (let [t (reduce + 0.0 (vals h))
        n (count h)]
    (reduce #(assoc %1 %2 (/ (* (/ 1.0 n) (h %2))
                             (/ t n)))
            {} (keys h))))


(defn bayes->joint-probability
  "Calculate joint probability. With the estimate posterior probability.
Type: [Double] -> Double -> Double"
  [probs priori]
  (let [a (apply * probs)
        b (apply * (map #(- 1.0 %) probs))]
    (/ (* a priori)  (+ (* a priori) (* b (- 1.0 priori))))))
