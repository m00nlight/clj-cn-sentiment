(ns clj-cn-sentiment.feature
  (require [clojure.string :as s]
           [clojure.java.io :as io]))


(defn- chi-sq
  [xs ys]
  (reduce (fn [acc [x1 x2]] (+ acc (/ (* (- x1 x2) (- x1 x2)) x2)))
          0.0 (map vector xs ys)))

(defn feature-selection
  "Select feature words based on their Chi-square score. Input is the content
of model file, and with an integer of n, return the top n feature words."
  [coll n]
  (let [total-pos (reduce + (map second coll))
        total-neg (reduce + (map #(nth % 2) coll))
        total-neu (reduce + (map last coll))
        exp-pos (/ (* 1.0 total-pos) (count coll))
        exp-neg (/ (* 1.0 total-neg) (count coll))
        exp-neu (/ (* 1.0 total-neu) (count coll))]
    (take n (sort-by #(- (chi-sq (drop 1 %) [exp-pos exp-neg exp-neu]))
                     (filter (fn [[word pos neg neu]]
                               (not (and (> pos 2000)
                                         (> neg 2000)
                                         (> neu 2000)))) coll)))))



;; (defn test-feature
;;   [n]
;;   (let [coll (map (fn [[word pos neg neu]]
;;                     (into [] [word (read-string pos) (read-string neg)
;;                               (read-string neu)]))
;;                   (map #(s/split % #"\t")
;;                        (s/split (slurp (io/resource "default.model")) #"\n")))]
;;     (feature-selection coll n)))
