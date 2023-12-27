//
//  File.swift
//  
//
//  Created by Caleb Jonas on 7/18/23.
//

import Foundation

func sub(_ x: [[Double]], _ y: [[Double]]) -> [[Double]] {
    let xrows = x.count
    let yrows = y.count
    let xcols = x.first!.count
    let ycols = y.first!.count
    precondition(xrows == yrows && xcols == ycols)

    return x.indices.map { row in
        x[row].indices.map { col in
            guard row < y.count && col < y.first!.count else { return 0 }
            return x[row][col] - y[row][col]
        }
    }
}

func sub(_ x: [[Double]], _ y: [Double]) -> [[Double]] {
    return x.indices.map { row in
        x[row].indices.map { col in
            return x[row][col] - y[col]
        }
    }
}

func sub(_ x: [[Double]], _ y: Double) -> [[Double]] {
    return x.indices.map { row in
        x[row].indices.map { col in
            return x[row][col] - y
        }
    }
}

func mmult(_ x: [[Double]], _ y: [[Double]]) -> [[Double]] {
    let rcols = x[0].count
    let rrows = x.count
    var result = [[Double]](repeating: [Double](repeating: 0, count: rcols), count: rrows)
    for i in 0..<x.count {
        for j in 0..<y[0].count {
            for k in 0..<x[0].count {
                result[i][j] += x[i][k] * y[k][j]
            }
        }
    }
    
    return result
}

func mdot(v1 x: [Double], v2 y: [Double]) -> [[Double]] {
    var result = [[Double]](repeating: [Double](repeating: 0, count: y.count), count: x.count)
    
    for i in 0..<result.count {
        for j in 0..<result[0].count {
            result[i][j] = x[i] * y[j]
        }
    }
    
    return result
}

func mdot(matrix: [[Double]], _ y: [Double]) -> [Double] {
    var result = [Double](repeating: 0, count: matrix[0].count)
    
    let columnWiseX = transpose(matrix)
    for i in 0..<columnWiseX.count {
        var sum: Double = 0
        for j in 0..<(columnWiseX[i].count - 1) {
            sum += columnWiseX[i][j]
        }
        result[i] = sum + columnWiseX[i].last! * y.reduce(1, *)
    }
    
    return result
}

func mmult(_ x: [[Double]], _ scalar: Double) -> [[Double]] {
    var matrix = x
    for i in 0..<x.count {
        for j in 0..<x[0].count {
            matrix[i][j] *= scalar
        }
    }
    
    return matrix
}

func softmax(_ x: [[Double]]) -> [[Double]] {
    let maxima = x.flatMap { $0 }.sorted(by: >).first!
    let ex_x = sub(x, maxima).map { $0.map {
        Double(exp($0))
    } }
    let sum = ex_x.flatMap { $0 }.reduce(0, +)
    let out = ex_x.map { arr in arr.map { $0 / sum } }
    return out
}

func reshape(_ m: [[Double]], rows: Int, columns: Int, repeating: Double = 0)  -> [[Double]] {
    var matrix = m
    
    matrix = matrix.dropLast(max(0, matrix.count - rows))
    if rows - matrix.count > 0 {
        matrix = matrix + [[Double]](repeating: [Double](repeating: repeating, count: columns), count: rows - matrix.count)
    }
    matrix = matrix.map {
        var x = $0
        x = x.dropLast(max(0, $0.count - columns))
        if columns - $0.count > 0 {
            x += [Double](repeating: repeating, count: columns - $0.count)
        }
        return x
    }
    
    return matrix
}

func printMatrix(_ matrix: [[Double]]) {
    (0..<matrix.count).forEach { idx in
        let row = matrix[idx]
        //            ｢ ⅂
        //            L ⅃
        if idx == 0 && matrix.count == 1 {
            print("[", terminator: "")
        } else if idx == 0 {
            print("｢", terminator: "")
        } else if idx == matrix.count - 1 {
            print("L", terminator: "")
        } else {
            print("|", terminator: "")
        }
        
        print("\t", terminator: "")
        
        print(row.map {
            String(format: "% 02.03f", $0)
        }.joined(separator: ", "), terminator: "")
        
        print("\t", terminator: "")
        
        if idx == 0 && matrix.count == 1 {
            print("]")
        } else if idx == 0 {
            print("⅂")
        } else if idx == matrix.count - 1 {
            print("⅃")
        } else {
            print("|")
        }
    }
    print("( rows= \(matrix.count), cols=\(matrix[0].count) )")
}
