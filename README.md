# NEST

NEST は C++で実装されたシンプルなインタプリタ型プログラミング言語です。

## ビルド方法

C++11 以降をサポートするコンパイラを使用して、インタプリタをコンパイルします。

```sh
g++ -std=c++11 -o nest nest.cc
```

## 使い方

`.nest` スクリプトファイルを指定して実行します。

```sh
./nest main.nest
```

## 言語仕様

### プログラム構造

すべての NEST プログラムは `nest main` ブロックで始まる必要があります。これがプログラムのエントリーポイントとなります。

```nest
nest main {
    // ここにコードを書きます
}
```

### 変数

`let` キーワードを使用して変数を宣言します。整数、文字列、ブール値をサポートしています。

```nest
let number = 42
let text = "Hello World"
```

### 出力

`print` 文を使用してコンソールに値を出力します。

```nest
print(text)
print(10 + 20)
```

### スコープ (Nest)

`nest` キーワードに続けて名前を指定することで、ネストされたスコープを作成できます。内側のスコープからは外側のスコープの変数にアクセスできますが、内側で宣言された変数はそのスコープ内でのみ有効です。

```nest
nest main {
    let global = "I am global"

    nest inner {
        print(global) // アクセス可能
        let local = "I am local"
    }

    // print(local) // エラー: localはこのスコープでは定義されていません
}
```

### 制御構文

#### If 文

条件分岐を行います。

```nest
let x = 10
if (x > 5) {
    print("x is greater than 5")
} else {
    print("x is small")
}
```

#### While ループ

条件が真である間、繰り返し実行します。

```nest
let i = 0
while (i < 3) {
    print(i)
    i = i + 1
}
```

#### For ループ

指定された範囲で繰り返し実行します。

```nest
for (i in 1..5) {
    print(i)
}
```

### 配列 (Arrays)

`[]` を使用して配列を作成し、インデックスでアクセスします。

```nest
let arr = [1, 2, 3]
print(arr[0]) // 1
```

### 辞書 (Dictionaries)

`{}` を使用してキーと値のペアを作成し、キーでアクセスします。

```nest
let dict = {"key": "value"}
print(dict["key"]) // value
```
