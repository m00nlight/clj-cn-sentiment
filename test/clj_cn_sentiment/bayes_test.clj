(ns clj-cn-sentiment.bayes-test
  (:require [clojure.test :refer :all]
            [clj-cn-sentiment.core :refer :all]
            [clj-cn-sentiment.bayes :refer :all]
            [clj-cn-sentiment.test-utils :as tu]))


(tu/with-private-fns [clj-cn-sentiment.bayes [count->probability]]
  (deftest test-private-count->probability
    (let [t1 (count->probability {:positive 1, :negative 1})
          t2 (count->probability {:positive 30, :negative 2})
          t3 (count->probability {:positive 2, :netural 300, :negative 30})]
      (testing "Testing for probabilities add up to 1.0"
        (is (< (Math/abs (- (apply + (vals t1)) 1.0)) 1e-6))
        (is (< (Math/abs (- (apply + (vals t2)) 1.0)) 1e-6))
        (is (< (Math/abs (- (apply + (vals t3)) 1.0)) 1e-6)))
      (testing "Testing for probabilities relationship"
        (is (< (:negative t2) (:positive t2)))
        (is (< (:positive t3) (:negative t3)))))))
