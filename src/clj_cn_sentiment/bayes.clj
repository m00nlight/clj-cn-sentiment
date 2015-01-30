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
