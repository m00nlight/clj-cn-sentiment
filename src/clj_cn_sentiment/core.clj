(ns clj-cn-sentiment.core
  (require [clojure.java.io :as io]
           [clojure.string :as s]
           [clj-cn-sentiment.bayes :as bayes]
           [clj-cn-sentiment.feature :as feature]
           [clj-cn-sentiment.segmentation :as seg]))

(def default-priori {:positive 1.0 :negative 1.0, :neutral 1.0})


(defn train
  "Train the Bayes classifier on positive and negative sentences. It take
two argument. The first one is the string indicate the localtion of the 
traing file, and the second one is the output location.

The training file should like the following:
positive        @       blablabla
negative        @       blablabla
positive        @       blablabla

Each line is an training sentence, and the first word is positive or negative,
indicate the sentence is a positive example or an negative example. Then 
follow an \t@\t then with the actual sentence."
  [^String training-file ^String output-path]
  (with-open [rdr (io/reader training-file)]
    (let [freq (reduce (fn [acc line]
                         (let [[pos-neg-neu sentence] (s/split line #"\t@\t")
                               words (seg/mmseg sentence)]
                           (reduce #(update-in
                                     %1 [%2 (keyword pos-neg-neu)]
                                     (fnil inc 0))
                                   acc words)))
                       {}
                       (line-seq rdr))]
      (with-open [wrdr (io/writer output-path :append true)]
        (doseq [word-info freq]
          (if (> (apply + (vals (second word-info))) 20)
            (.write wrdr (str (first word-info) "\t"
                              (:positive (second word-info) 0)
                              "\t"
                              (:negative (second word-info) 0)
                              "\t"
                              (:neutral (second word-info) 0)
                              "\n"))))))))

(defn load-model
  "Load the training model for usage. Rate is the rate of positive example 
over the negative example in the training data. It is used for training on
an corpus."
  ([n] (load-model n "default.model"))
  ([n model-file]
     (let [temp (map #(s/split % #"\t")
                     (s/split (slurp (if (= model-file "default.model")
                                       (io/resource "default.model")
                                       (io/file model-file)))
                              #"\n"))
           ;; filter out one character words with occurance in positive and
           ;; negative are both more than 2000 times, it is maybe an normal
           ;; words, not useful for classify
           total-pos (reduce + (map #(read-string (second %)) temp))
           total-neg (reduce + (map #(read-string (nth % 2)) temp))
           total-neu (reduce + (map #(read-string (last %)) temp))
           freq (feature/feature-selection
                 (map (fn [[word pos neg neu]]
                        (into [] [word (read-string pos)
                                  (read-string neg)
                                  (read-string neu)])) temp) n)]
       (reduce #(assoc %1 (first %2)
                       (-> {:positive (max (/ (second %2) total-pos) 1e-6)
                            :negative (max (/ (nth %2 2) total-neg) 1e-6)
                            :neutral (max (/ (last %2) total-neu) 1e-6)}
                           bayes/count->probability))
               {}
               freq))))


(def default-model (load-model 50000))

(defn- get-prob-helper
  "Get the probability, if the probability is an negative phrase, and 
the negative phrase is not in the model, we inverse the the phrase with
the negative words dropped.

For example: 不痛快 does not appear in the model, so we inverse the 
probability of 痛快 as the probability of 不痛快."
  [word model]
  (cond
   ;; the word in the model
   (not (nil? (get model word))) (get model word)
   ;; the word is not in the model, and start with an negative character
   (contains? seg/cn-not-words (str (first word)))
   (let [w (apply str (rest word))]
     (if-let [m (get model w)]
       {:positive (- 1.0 (+ (:neutral m) (:negative m)))
        :neutral (- 1.0 (+ (:positive m) (:negative m)))
        :negative (- 1.0 (+ (:positive m) (:neutral m)))}
       default-priori))
   ;; else, not in the model and the word is not start with negative character
   :else default-priori))

(defn classify
  "Classify the input text, and give the probability of pos and neg."
  ([text] (classify text default-model default-priori))
  ([text model priori]
     (let [words (seg/mmseg text ) 
           probs (sort-by #(- (get-in % [:prob :negative]))
                          (map (fn [x] {:word x,
                                        :prob (get-prob-helper x model)})
                               words))
           nums (max 4 (int (* 0.15 (count probs))))
           ret-neg (bayes/bayes->probability-distribution
                    (take nums probs)
                    priori)
           ret-pos (bayes/bayes->probability-distribution
                    (take nums (sort-by #(- (get-in % [:prob :positive]))
                                       probs))
                    priori)]
       (bayes/bayes->probability-distribution
        probs
        priori)
       ;; (if (> (:positive ret-pos) (:negative ret-neg))
       ;;   ret-pos
       ;;   ret-neg)
       )))


(defn evaluate-pos-neg
  "Evaluate pos and neg precision and recall"
  [n file]
  (let [model (load-model n)
        p1 (atom 0)
        p2 (atom 0)
        p3 (atom 0)
        n1 (atom 0)
        n2 (atom 0)
        n3 (atom 0)]
    (with-open [rdr (io/reader file)]
      (doseq [line (line-seq rdr)]
        (let [[score text] (s/split line #"\|")
              {pos :positive, neg :negative, neu :neutral}
              (classify text model default-priori)]
          (cond
           (= score "1")
           (if (> pos neg)
             (do
               (swap! p1 inc)
               (swap! p2 inc)
               (swap! p3 inc))
             (do
               (swap! p2 inc)
               (swap! n3 inc)))
           (= score "-1")
           (if (> neg pos)
             (do
               (swap! n1 inc)
               (swap! n2 inc)
               (swap! n3 inc))
             (do
               (swap! n2 inc)
               (swap! p3 inc))))))
      (let [p-pre (/ @p1 (* 1.0 @p3))
            p-rec (/ @p1 (* 1.0 @p2))
            n-pre (/ @n1 (* 1.0 @n3))
            n-rec (/ @n1 (* 1.0 @n2))
            p-f1 (/ (* 2 p-pre p-rec) (+ p-pre p-rec))
            n-f1 (/ (* 2 n-pre n-rec) (+ n-pre n-rec))]
        (println (str "positive precision: " p-pre))
        (println (str "positive recall   : " p-rec))
        (println (str "positive f1-score : " p-f1))
        (println (str "negative precision: " n-pre))
        (println (str "negative recall   : " n-rec))
        (println (str "negative f1-score : " n-f1))))))

(defn evaluate-pos-neg-neu
  [n file]
  (let [model (load-model n)
        p1 (atom 0)
        p2 (atom 0)
        p3 (atom 0)
        n1 (atom 0)
        n2 (atom 0)
        n3 (atom 0)
        nn1 (atom 0)
        nn2 (atom 0)
        nn3 (atom 0)]
    (with-open [rdr (io/reader file)]
      (doseq [line (line-seq rdr)]
        (let [[score text] (s/split line #"\|")
              {pos :positive, neg :negative, neu :neutral}
              (classify text model default-priori)]
          (cond
           (= score "1")
           (if (and (> pos neg) (> pos neu))
             (do
               (swap! p1 inc)
               (swap! p2 inc)
               (swap! p3 inc))
             (cond
              (and (> neg pos) (> neg neu)) (do
                                              (swap! p2 inc)
                                              (swap! n3 inc))
              :else (do (swap! p2 inc) (swap! nn3 inc))))
           (= score "-1")
           (if (and (> neg pos) (> neg neu))
             (do
               (swap! n1 inc)
               (swap! n2 inc)
               (swap! n3 inc))
             (cond
              (and (> pos neg) (> pos neu)) (do (swap! n2 inc) (swap! p3 inc))
              :else (do (swap! n2 inc) (swap! nn3 inc))))
           (= score "0")
           (if (and (> neu pos) (> neu neg))
             (do
               (swap! nn1 inc)
               (swap! nn2 inc)
               (swap! nn3 inc))
             (cond
              (and (> pos neg) (> pos neu)) (do (swap! nn2 inc) (swap! p3 inc))
              :else (do (swap! nn2 inc) (swap! n3 inc)))))))
      (let [p-pre (/ @p1 (* 1.0 @p3))
            p-rec (/ @p1 (* 1.0 @p2))
            n-pre (/ @n1 (* 1.0 @n3))
            n-rec (/ @n1 (* 1.0 @n2))
            nn-pre (/ @nn1 (* 1.0 @nn2))
            nn-rec (/ @nn1 (* 1.0 @nn3))
            p-f1 (/ (* 2 p-pre p-rec) (+ p-pre p-rec))
            n-f1 (/ (* 2 n-pre n-rec) (+ n-pre n-rec))
            nn-f1 (/ (* 2 nn-pre nn-rec) (+ nn-pre nn-rec))]
        (println (str "positive precision: " p-pre))
        (println (str "positive recall   : " p-rec))
        (println (str "positive f1-score : " p-f1))
        (println (str "negative precision: " n-pre))
        (println (str "negative recall   : " n-rec))
        (println (str "negative f1-score : " n-f1))
        (println (str "neutral precision : " nn-pre))
        (println (str "neutral recall    : " nn-rec))
        (println (str "neutral f1-score  : " nn-f1))))))


(defn main
  []
  (doseq [n [100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
             20000 30000 40000 70000]]
    (println "feature words numbers: " n)
    (evaluate-pos-neg
     n
     "/home/m00nlight/Documents/work/sentiment/input-tweet.txt")
    (println "=====================================================")))


(defn main2
  []
  (doseq [n [100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
             20000 30000 40000 70000]]
    (println "feature words numbers: " n)
    (evaluate-pos-neg-neu
     n
     "/home/m00nlight/Documents/work/sentiment/input-tweet.txt")
    (println "=====================================================")))
