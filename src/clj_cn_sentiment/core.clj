(ns clj-cn-sentiment.core
  (require [clojure.java.io :as io]
           [clojure.string :as s]
           [clj-cn-sentiment.bayes :as bayes]
           [clj-cn-sentiment.segmentation :as seg]))

(def default-priori {:positive 0.728 :negative 0.2719})

(def default-rate (/ (:positive default-priori) (:negative default-priori)))

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
                         (let [[pos-or-neg sentence] (s/split line #"\t@\t")
                               words (seg/mmseg sentence)]
                           (reduce #(update-in
                                     %1 [%2 (keyword pos-or-neg)] (fnil inc 0))
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
                              "\n"))))))))

(defn load-model
  "Load the training model for usage. Rate is the rate of positive example 
over the negative example in the training data. It is used for training on
an balanced corpus. The default-rate is the rate of value traing the default
model."
  ([] (load-model "default.model" default-rate))
  ([model-file rate]
     (let [freq (s/split (slurp (if (= model-file "default.model")
                                  (io/resource "default.model")
                                  (io/file model-file)))
                         #"\n")]
       (reduce #(assoc %1 (first %2)
                       (-> {:positive (/ (read-string (second %2)) rate)
                            :negative (read-string (last %2))}
                           bayes/count->probability) )
               {}
               (map #(s/split % #"\t") freq)))))


(def default-model nil)

(defn classify
  "Classify the input text, and give the probability of pos and neg."
  ([text] (classify text default-model default-priori))
  ([text model priori]
     (let [words (seg/mmseg text)
           probs (sort-by #(- (get-in % [:prob :negative]))
                          (map (fn [x] {:word x,
                                        :prob (get model x {:positive 0.5
                                                            :negative 0.5})})
                               words))]
       )))
