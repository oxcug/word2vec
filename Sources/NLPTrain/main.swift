import Foundation
import NLPPOS

let neuronsInHiddenLayerCount = 3

let stopWords = ["you","something","anyhow","would","not","first","now","without","which","may","regarding","'d","back","nevertheless","how","should","bottom","by","twelve","least","but","‘d","thence","i","hers","are","therein","same","indeed","others","whither","your","'ll","either","last","therefore","do","whence","we","top","beforehand","though","across","everyone","only","full","fifteen","hereby","since","while","re","beside","quite","her","is","their","meanwhile","neither","various","everywhere", "'d","made","nowhere","name","of","done","ever","onto","off","its","most","twenty","next","after","does","whether","say","please","at","sometimes", "n't","hereafter","here","until","itself","latterly","well","became","under","behind","the","me","must","give","former","using","or","otherwise","noone","‘s","yours","everything","wherein","even","take","put","ourselves","themselves","him","beyond","whose","another","with","every","whom","somewhere","forty","via","'ve","get", "'s","‘re","any","due","really","'re","towards","it","whereupon","none","anyway","very","among","before","sixty","eleven","seeming","why","whereby","whenever","per","ours","namely","they", "'m","along","somehow","yourself","many","empty","who","becoming","hence","them","n't","between","a","be","further","against","else","when","has","will","anyone","was","several","there","three","formerly","one","my","were","side","cannot","becomes", "'ll","make","such","never","amount","enough","just","our","those","besides","'s","being","part","except","someone","often","seems","‘ve","latter", "'ve","afterwards","both","during","unless","together","n‘t","show","keep","too","each","into","been","an","us","whereafter","to","in","nor","‘ll","so", "'re","down","six","toward","five","doing","out","herein","thereupon","whole","anything","can","because","over","however","seem","serious","go","am","then","myself","within","four","his","nobody","sometime","yet","front","become","himself","wherever","upon","nothing","few","hundred","move","‘m","what","as","below","elsewhere","mostly","anywhere","up","that","amongst","this","she","always","thereafter","nine","ca","already","herself","some","much","if","two","these","had","ten","whatever","also","through","thus","yourselves","see","he","throughout","for","around","moreover","’m","seemed","again","might","all","on","almost","have","less","fifty","eight","could","used","thereby","perhaps","above","whereas","and","about","although","still","mine","from","than","rather","once","third","call","alone","did","more","thru","whoever","where","hereupon","other","own","no"]

let descTrainingData = """
The wide road shimmered in the hot sun
The wide road shimmered in the hot sun
The wide road shimmered in the hot sun
The wide road shimmered in the hot sun
"""

let test = "road"


extension [Double] {
    static func &(lhs: [Double], rhs: [Double]) -> [Double] {
        (0..<lhs.count).map { Swift.max(lhs[$0], rhs[$0]) }
    }
}

func createTrainingData(with ohe: OneHotEncoder) -> (x: [[Double]] , t: [[Double]]) {
    let sortedWordEncs = ohe.sortedWordEncodingsByOccurence
    let t = sortedWordEncs.map { word in
        (word.behind + word.ahead).reduce(into: word.encoding.map { _ in 0.0 }) { $0 = $0 & ohe.wordEncoding(forIndex: $1)!.encoding }
    }
    let x = sortedWordEncs.map { $0.encoding }
    return (x: x, t: t)
}

//func getPartsOfSpeechTrainingDataSet() -> (test: String, trainingData: String) {
//    let dataDir = URL(string: #file)!.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("NLPPOS/words-by-partsOfSpeech/words-by-partsOfSpeech")
//    let contents = try! FileManager.default.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: [.isRegularFileKey])
//    
//    var x = contents.flatMap { url in
//        let pos = url.deletingPathExtension().lastPathComponent
//        let data = try! String(contentsOf: url, encoding: .utf8)
//        return data.lowercased().replacingOccurrences(of: "!", with: "").split(separator: "\r\n").compactMap {
//            guard !$0.isEmpty && !String($0).contains(" ") else { return Optional<String>(nil) }
//            return "\($0)|\(pos.lowercased())"
//        }
//    }
//    
//    var g = SystemRandomNumberGenerator()
//    x.shuffle(using: &g)
//    let out = Array(x.prefix(100))
//    let t = out.randomElement()!.components(separatedBy: "|").first!
//    return (test: t, trainingData: out.joined(separator: ","))
//}

func train() -> NNModel! {
    let vocab = OneHotEncoder(descTrainingData, slidingWindow: 1)
    let trainingData = createTrainingData(with: vocab)
    
    precondition(neuronsInHiddenLayerCount <= vocab.count, "\(neuronsInHiddenLayerCount) is not <= \(vocab.count)")
    
    // we now have a one hot encoded input layer and the bitsize, now we should allocate our n-net.
    var r = SystemRandomNumberGenerator()
    func randomWeightMatrix(rows: Int, columns: Int) -> [[Double]] {
        [[Double]](repeating: [Double](repeating: 0, count: columns), count: rows).map { $0.map { _ in .random(in: -0.8..<0.8, using: &r) } }
    }
    var nnet = NNModel(
        w1: randomWeightMatrix(rows: vocab.count, columns: neuronsInHiddenLayerCount),
        w2: randomWeightMatrix(rows: neuronsInHiddenLayerCount, columns: vocab.count)
    )
    
    let epoch = 20000
    var alpha = 0.001
    
    print("Model Weights1:")
    printMatrix(nnet.w1)
    print("Model Weights2:")
    printMatrix(nnet.w2)

    var loss: Double = 0
    for i in 1...epoch {
        print("epoch #\(i) - alpha = \(alpha) - loss = \(loss)")
    
        (0..<trainingData.x.count).forEach { idx in
            let x = trainingData.x[idx]
            let t = trainingData.t[idx]
			let fwd = Word2Vec.SkipGram.feedfwd(model: &nnet, input: x, vocabSize: vocab.count)
			Word2Vec.SkipGram.backprop(forward: fwd, model: &nnet, input: x, training: t, vocabSize: vocab.count, alpha: alpha)
            guard t[idx] == 1 else { return }
            
//            loss += -1 * fwd.embeddings2[idx][0]
//            loss += 1
//            loss += C * np.log(np.sum(np.exp(self.u)))
        }
        
        alpha *= 1.0 / (1.0 + alpha * Double(i))
    }
    
    struct R {
        var idx: Int
        var key: String
        var confidence: Double
    }
    
	let p = Word2Vec.SkipGram.feedfwd(model: &nnet, input: vocab.wordEncoding(forWord: test)!.encoding, vocabSize: vocab.count).z
    func printPrediction(for r: R) {
        print("\(r.idx) = \"\(r.key)\" (confidence \(String(format: "%2.2f", 100 * r.confidence))%)")
    }
    print("Model Weights1:")
    printMatrix(nnet.w1)
    print("Model Weights2:")
    printMatrix(nnet.w2)
    print("Predictions = \(p)")
    print("Test = '\(test)'")
    
    var rs = (0..<vocab.count).map { idx in
        let word = vocab.word(forIndex: idx)!
        let elm = vocab.wordEncoding(forIndex: idx)!
        let r = R(idx: elm.idx, key: word, confidence: p[elm.idx][0])
        return r
    }
    
    rs.sort(by: { $0.confidence > $1.confidence })
    rs.dropLast(abs(3 - rs.count)).forEach {
        printPrediction(for: $0)
    }
    
    return nnet
}

struct FeedForwardResponse {
    var z: [[Double]]
    var embeddings1: [Double]
    var embeddings2: [Double]
}

func transpose(_ matrix: [[Double]]) -> [[Double]] {
    let rcols = matrix.count
    let rrows = matrix.first!.count
    var result = [[Double]](repeating: [Double](repeating: 0, count: rcols), count: rrows)
    (0..<rrows).forEach { row in
        (0..<rcols).forEach { col in
            result[row][col] = matrix[col][row]
        }
    }
    return result
}

//let a: matrix_float2x4 = .init([1.0, 2.0, 3, 4], [5.0, 6, 7, 8.0])
//let b: matrix_float2x2 = .init([5,6], [7,8])
//print("[[1, 2, 3, 4], [5, 6, 7, 8]] * [[5,6], [7,8]]")
//print(a * b)
//
//let x: matrix_float2x4 = .init([1.0, 2.0, 3, 4], [5.0, 6, 7, 8.0])
//let y: simd_float2 = .init(x: 1, y: 2)
//
//print("[[1, 2, 3, 4], [5, 6, 7, 8]] * [1, 2]")
//print(x * y)
//printMatrix(mdot([[1.0, 2.0, 3, 4], [5.0, 6, 7, 8.0]], [1, 2]))
//
//printMatrix(mmult(
//   [[1, 2, 3, 4],
//    [5, 6, 7, 8]],
//   [[5,6],
//    [7,8]]))

_ = train()
