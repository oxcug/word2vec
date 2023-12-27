//
//  Word2Vec.swift
//  
//
//  Created by Caleb Jonas on 8/3/23.
//

import Foundation

protocol NeuralNetAlgorithm {
    static func feedfwd(model: inout NNModel, input: [Double], vocabSize: Int) -> FeedForwardResponse
    static func backprop(forward fwd: FeedForwardResponse,  model: inout NNModel, input: [Double], training: [Double], vocabSize: Int, alpha: Double)
}

struct Word2Vec  {
    
    struct SkipGram: NeuralNetAlgorithm {
        
        static func feedfwd(model: inout NNModel, input: [Double], vocabSize: Int) -> FeedForwardResponse {
            let w1 = model.w1
            let w2 = model.w2
            let e1 = mdot(matrix: w1, input)
            let e2 = mdot(matrix: w2, e1)
            let e2x = transpose([e2])
            let z = softmax(e2x)
            return FeedForwardResponse(z: z, embeddings1: e1, embeddings2: e2)
        }
        
        static func backprop(forward fwd: FeedForwardResponse,  model: inout NNModel, input: [Double], training: [Double], vocabSize: Int, alpha: Double = 0.01) {
            let tmp = transpose([training])
            let e = sub(fwd.z, tmp)
            let bw2 = mdot(v1: fwd.embeddings1, v2: transpose(e)[0])
            
            let X = transpose([input])
            let _tmp_bw1 = transpose(mmult(model.w1, e))
            let bw1 = mmult(_tmp_bw1, X)
            
            // adjust weights
            let _bw2 = mmult(bw2, alpha)
            model.w2 = sub(model.w2, _bw2)
            let _bw1 = mmult(bw1, alpha)
            model.w1 = sub(model.w1, transpose(_bw1))
        }
    }
}
