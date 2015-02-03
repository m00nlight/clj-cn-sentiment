(ns clj-cn-sentiment.segmentation
  (require [clj-cn-mmseg.core :as mmseg]))



(def cn-not-words
  #{"不", "没", "无", "非", "莫", "弗", "毋", "勿", "未", "否", "别", "無", "休"
    ""})


(defn combine-not-words
  "Combine not words with the next words if "
  [coll]
  (letfn [(helper [coll acc]
            (cond
             (empty? coll) (reverse acc)
             ;; current word is not word and the next word is an adjective
             (and (contains? cn-not-words (:word (first coll)))
                  (not (nil? (first (rest coll))))
                  (>= (count (dissoc (:nature (first coll))
                                     :split-only :per :loc)) 1))
             (let [[x y & z] coll]
               (recur z (cons {:word (str (:word x) (:word y)) :not x :adj y
                               :nature {:neg-phrase true}}
                              acc)))
             ;; else, the current node is not the negative word or the
             ;; the negative words is not follow by an adjective.
             :else (recur (rest coll) (cons (first coll) acc))))]
    (helper coll '())))

(defn mmseg
  [text]
  "Segmentation the sentence."
  (map :word (filter #(and (not (empty? (% :nature)))
                           (not (get-in % [:nature :unit] false))
                           (not (get-in % [:nature :concurrency] false))
                           (not (get-in % [:nature :num] false)))
                     (-> text mmseg/mmseg combine-not-words))))

