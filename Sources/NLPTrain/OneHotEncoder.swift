//
//  File.swift
//  
//
//  Created by Caleb Jonas on 7/18/23.
//

import Foundation

class OneHotEncoder {
    
    struct EncodedWord {
        var behind: [Int]
        var ahead: [Int]
        var idx: Int
        var occurences: Int
        var encoding: [Double]
    }
    
    private var vocabMap = [String:Int]()
    private var vocabReverseMap = [Int:String]()
    private var vocabVec = [EncodedWord]()
    
    var count: Int { vocabMap.count }
    
    var sortedWordEncodingsByOccurence: [EncodedWord] { vocabVec }
    
    func wordEncoding(forWord word: String) -> EncodedWord? {
        vocabVec[vocabMap[word]!]
    }
    
    func wordEncoding(forIndex index: Int) -> EncodedWord? {
        vocabVec[vocabMap[vocabReverseMap[index]!]!]
    }
    
    func word(forIndex index: Int) -> String? {
        vocabReverseMap[index]
    }
    
    init(_ dataset: String, slidingWindow: UInt) {
        let scentences = dataset
            .replacingOccurrences(of: "\n", with: " ")
            .split(separator: ".")
            .map {
                $0.split(separator: " ")
                    .map { $0.lowercased() }
                    .map { String($0) }
                    .filter { !stopWords.contains($0) }
            }
        
        let allWords = scentences.indices.flatMap { j in
            let offset = (0..<max(0, j)).reduce(into: 0) { $0 += scentences[$1].count }
            let s_words = scentences[j]
            return s_words.indices.map { i in
                let s = Int(slidingWindow)
                let w = s_words[i]
                let behind = (max(-s, -s + s - i)..<0).map { offset + $0 + i }
                let ahead = (min(s_words.count, i + 1)..<min(s_words.count, i + s + 1)).map { offset + $0 }
//                print()
                return (key: w, occurences: s_words.filter({ $0 == w }).count, behind: behind, ahead: ahead, scentence: j)
            }
        }
            .sorted { $0.occurences > $1.occurences }
        
        allWords.forEach { w in
            guard vocabMap[w.0] != nil else {
                let idx = self.vocabVec.count
                let enc = EncodedWord(behind: w.behind.map { $0 }, ahead: w.ahead.map { $0 }, idx: idx, occurences: w.1, encoding: [])
                vocabMap[w.0] = idx
                vocabReverseMap[idx] = w.0
                vocabVec.append(enc)
                return
            }
        }
        
        vocabVec.indices.forEach { i in
            vocabVec[i].behind = vocabVec[i].behind.map { vocabMap[allWords[$0].key]! }
            vocabVec[i].ahead = vocabVec[i].ahead.map { vocabMap[allWords[$0].key]! }
        }
            
        (0..<vocabVec.count).forEach { i in
            var encoding = [Double](repeating: 0, count: vocabVec.count)
            encoding[vocabVec[i].idx] = 1
            vocabVec[i].encoding = encoding
        }
    }
}
