#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <cctype>

// --- 1. Tokenizer (Lexer) ---

enum TokenType {
    TOKEN_EOF,
    KEYWORD_NEST, KEYWORD_LET, KEYWORD_IF, KEYWORD_ELSE,
    KEYWORD_WHILE, KEYWORD_FOR, KEYWORD_IN, KEYWORD_PRINT,
    IDENTIFIER, NUMBER, STRING_LITERAL,
    PLUS, MINUS, STAR, SLASH, PERCENT,
    LT, LTE, GT, GTE, EQ, NEQ,
    ASSIGN, LBRACE, RBRACE, LPAREN, RPAREN, DOTDOT,
    LBRACKET, RBRACKET, COLON, COMMA
};

struct Token {
    TokenType type;
    std::string text;
    int line;
};

class Lexer {
    std::string src;
    size_t pos = 0;
    int line = 1;

public:
    Lexer(const std::string& source) : src(source) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (pos < src.length()) {
            char current = src[pos];

            if (std::isspace(current)) {
                if (current == '\n') line++;
                pos++;
            } else if (std::isdigit(current)) {
                tokens.push_back(number());
            } else if (std::isalpha(current)) {
                tokens.push_back(identifier());
            } else if (current == '"') {
                tokens.push_back(stringLiteral());
            } else {
                switch (current) {
                    case '+': tokens.push_back({PLUS, "+", line}); pos++; break;
                    case '-': tokens.push_back({MINUS, "-", line}); pos++; break;
                    case '*': tokens.push_back({STAR, "*", line}); pos++; break;
                    case '/': tokens.push_back({SLASH, "/", line}); pos++; break;
                    case '%': tokens.push_back({PERCENT, "%", line}); pos++; break;
                    case '(': tokens.push_back({LPAREN, "(", line}); pos++; break;
                    case ')': tokens.push_back({RPAREN, ")", line}); pos++; break;
                    case '{': tokens.push_back({LBRACE, "{", line}); pos++; break;
                    case '}': tokens.push_back({RBRACE, "}", line}); pos++; break;
                    case '[': tokens.push_back({LBRACKET, "[", line}); pos++; break;
                    case ']': tokens.push_back({RBRACKET, "]", line}); pos++; break;
                    case ':': tokens.push_back({COLON, ":", line}); pos++; break;
                    case ',': tokens.push_back({COMMA, ",", line}); pos++; break;
                    case '=':
                        if (peek() == '=') { tokens.push_back({EQ, "==", line}); pos+=2; }
                        else { tokens.push_back({ASSIGN, "=", line}); pos++; }
                        break;
                    case '!':
                        if (peek() == '=') { tokens.push_back({NEQ, "!=", line}); pos+=2; }
                        else { throw std::runtime_error("Unexpected character '!' at line " + std::to_string(line)); }
                        break;
                    case '<':
                        if (peek() == '=') { tokens.push_back({LTE, "<=", line}); pos+=2; }
                        else { tokens.push_back({LT, "<", line}); pos++; }
                        break;
                    case '>':
                        if (peek() == '=') { tokens.push_back({GTE, ">=", line}); pos+=2; }
                        else { tokens.push_back({GT, ">", line}); pos++; }
                        break;
                    case '.':
                        if (peek() == '.') { tokens.push_back({DOTDOT, "..", line}); pos+=2; }
                        else { throw std::runtime_error("Unexpected character '.' at line " + std::to_string(line)); }
                        break;
                    default:
                        throw std::runtime_error(std::string("Unknown character: ") + current + " at line " + std::to_string(line));
                }
            }
        }
        tokens.push_back({TOKEN_EOF, "", line});
        return tokens;
    }

private:
    char peek() { return (pos + 1 < src.length()) ? src[pos + 1] : '\0'; }

    Token number() {
        size_t start = pos;
        while (pos < src.length() && std::isdigit(src[pos])) pos++;
        return {NUMBER, src.substr(start, pos - start), line};
    }

    Token stringLiteral() {
        pos++; // skip opening quote
        size_t start = pos;
        while (pos < src.length() && src[pos] != '"') pos++;
        std::string content = src.substr(start, pos - start);
        if (pos < src.length()) pos++; // skip closing quote
        return {STRING_LITERAL, content, line};
    }

    Token identifier() {
        size_t start = pos;
        while (pos < src.length() && (std::isalnum(src[pos]) || src[pos] == '_')) pos++;
        std::string text = src.substr(start, pos - start);
        TokenType type = IDENTIFIER;
        if (text == "nest") type = KEYWORD_NEST;
        else if (text == "let") type = KEYWORD_LET;
        else if (text == "if") type = KEYWORD_IF;
        else if (text == "else") type = KEYWORD_ELSE;
        else if (text == "while") type = KEYWORD_WHILE;
        else if (text == "for") type = KEYWORD_FOR;
        else if (text == "in") type = KEYWORD_IN;
        else if (text == "print") type = KEYWORD_PRINT;
        return {type, text, line};
    }
};

// --- 2. AST & Runtime Values (C++11 Compatible) ---

enum ValueType { VAL_INT, VAL_STR, VAL_BOOL, VAL_ARRAY, VAL_DICT };

struct Value {
    ValueType type;
    int iVal;
    std::string sVal;
    bool bVal;
    std::vector<Value> arrayVal;
    std::map<std::string, Value> dictVal;

    Value() : type(VAL_INT), iVal(0), bVal(false) {}
    Value(int v) : type(VAL_INT), iVal(v), bVal(false) {}
    Value(std::string v) : type(VAL_STR), iVal(0), sVal(v), bVal(false) {}
    Value(bool v) : type(VAL_BOOL), iVal(0), bVal(v) {}
    Value(std::vector<Value> v) : type(VAL_ARRAY), iVal(0), bVal(false), arrayVal(v) {}
    Value(std::map<std::string, Value> v) : type(VAL_DICT), iVal(0), bVal(false), dictVal(v) {}
    // C strings
    Value(const char* v) : type(VAL_STR), iVal(0), sVal(v), bVal(false) {}

    std::string toString() const {
        if (type == VAL_INT) return std::to_string(iVal);
        if (type == VAL_STR) return sVal;
        if (type == VAL_BOOL) return bVal ? "true" : "false";
        if (type == VAL_ARRAY) {
            std::string res = "[";
            for (size_t i = 0; i < arrayVal.size(); ++i) {
                if (i > 0) res += ", ";
                res += arrayVal[i].toString();
            }
            res += "]";
            return res;
        }
        if (type == VAL_DICT) {
            std::string res = "{";
            bool first = true;
            for (const auto& kv : dictVal) {
                if (!first) res += ", ";
                res += "\"" + kv.first + "\": " + kv.second.toString();
                first = false;
            }
            res += "}";
            return res;
        }
        return "";
    }

    // Operator overloading
    Value operator+(const Value& other) const {
        if (type == VAL_STR || other.type == VAL_STR) {
            return Value(toString() + other.toString());
        }
        return Value(iVal + other.iVal);
    }

    Value operator-(const Value& o) const { return Value(iVal - o.iVal); }
    Value operator*(const Value& o) const { return Value(iVal * o.iVal); }
    Value operator/(const Value& o) const { return Value(iVal / o.iVal); }
    Value operator%(const Value& o) const { return Value(iVal % o.iVal); }

    bool operator<(const Value& o) const { return iVal < o.iVal; }
    bool operator<=(const Value& o) const { return iVal <= o.iVal; }
    bool operator>(const Value& o) const { return iVal > o.iVal; }
    bool operator>=(const Value& o) const { return iVal >= o.iVal; }

    bool operator==(const Value& o) const {
        if (type != o.type) return false;
        if (type == VAL_INT) return iVal == o.iVal;
        if (type == VAL_STR) return sVal == o.sVal;
        if (type == VAL_BOOL) return bVal == o.bVal;
        if (type == VAL_ARRAY) return arrayVal == o.arrayVal;
        if (type == VAL_DICT) return dictVal == o.dictVal;
        return false;
    }
    bool operator!=(const Value& o) const { return !(*this == o); }

    bool isTruthy() const {
        if (type == VAL_BOOL) return bVal;
        if (type == VAL_INT) return iVal != 0;
        return !sVal.empty();
    }
};

class Environment {
    std::map<std::string, Value> values;
    Environment* parent = nullptr;

public:
    Environment(Environment* p = nullptr) : parent(p) {}

    void define(const std::string& name, Value val) {
        values[name] = val;
    }

    void assign(const std::string& name, Value val) {
        if (values.count(name)) {
            values[name] = val;
            return;
        }
        if (parent) {
            parent->assign(name, val);
            return;
        }
        throw std::runtime_error("Undefined variable '" + name + "'");
    }

    Value get(const std::string& name) {
        if (values.count(name)) return values[name];
        if (parent) return parent->get(name);
        throw std::runtime_error("Undefined variable '" + name + "'");
    }
};

// AST Nodes
struct Expr {
    virtual ~Expr() = default;
    virtual Value evaluate(Environment& env) = 0;
};

struct Stmt {
    virtual ~Stmt() = default;
    virtual void execute(Environment& env) = 0;
};

// Expression Implementations
struct LiteralExpr : Expr {
    Value value;
    LiteralExpr(Value v) : value(v) {}
    Value evaluate(Environment&) override { return value; }
};

struct VariableExpr : Expr {
    std::string name;
    VariableExpr(std::string n) : name(n) {}
    Value evaluate(Environment& env) override { return env.get(name); }
};

struct BinaryExpr : Expr {
    std::shared_ptr<Expr> left, right;
    TokenType op;
    BinaryExpr(std::shared_ptr<Expr> l, TokenType o, std::shared_ptr<Expr> r) : left(l), op(o), right(r) {}

    Value evaluate(Environment& env) override {
        Value l = left->evaluate(env);
        Value r = right->evaluate(env);
        switch (op) {
            case PLUS: return l + r;
            case MINUS: return l - r;
            case STAR: return l * r;
            case SLASH: return l / r;
            case PERCENT: return l % r;
            case LT: return l < r;
            case LTE: return l <= r;
            case GT: return l > r;
            case GTE: return l >= r;
            case EQ: return l == r;
            case NEQ: return l != r;
            default: throw std::runtime_error("Unknown operator");
        }
    }
};

struct ArrayExpr : Expr {
    std::vector<std::shared_ptr<Expr>> elements;
    ArrayExpr(std::vector<std::shared_ptr<Expr>> e) : elements(e) {}
    Value evaluate(Environment& env) override {
        std::vector<Value> values;
        for (auto& expr : elements) {
            values.push_back(expr->evaluate(env));
        }
        return Value(values);
    }
};

struct DictExpr : Expr {
    std::map<std::string, std::shared_ptr<Expr>> pairs;
    DictExpr(std::map<std::string, std::shared_ptr<Expr>> p) : pairs(p) {}
    Value evaluate(Environment& env) override {
        std::map<std::string, Value> values;
        for (auto& kv : pairs) {
            values[kv.first] = kv.second->evaluate(env);
        }
        return Value(values);
    }
};

struct IndexExpr : Expr {
    std::shared_ptr<Expr> object;
    std::shared_ptr<Expr> index;
    IndexExpr(std::shared_ptr<Expr> o, std::shared_ptr<Expr> i) : object(o), index(i) {}

    Value evaluate(Environment& env) override {
        Value obj = object->evaluate(env);
        Value idx = index->evaluate(env);

        if (obj.type == VAL_ARRAY) {
            if (idx.type != VAL_INT) throw std::runtime_error("Array index must be an integer.");
            if (idx.iVal < 0 || idx.iVal >= obj.arrayVal.size()) throw std::runtime_error("Array index out of bounds.");
            return obj.arrayVal[idx.iVal];
        }
        if (obj.type == VAL_DICT) {
            if (idx.type != VAL_STR) throw std::runtime_error("Dict key must be a string.");
            if (obj.dictVal.count(idx.sVal)) return obj.dictVal[idx.sVal];
            throw std::runtime_error("Key not found: " + idx.sVal);
        }
        throw std::runtime_error("Cannot index non-array/dict value.");
    }
};

// Statement Implementations
struct BlockStmt : Stmt {
    std::vector<std::shared_ptr<Stmt>> statements;
    void execute(Environment& env) override {
        for (auto& stmt : statements) stmt->execute(env);
    }
};

struct NestStmt : Stmt {
    std::string name;
    std::shared_ptr<BlockStmt> body;
    NestStmt(std::string n, std::shared_ptr<BlockStmt> b) : name(n), body(b) {}

    void execute(Environment& env) override {
        Environment newEnv(&env);
        body->execute(newEnv);
    }
};

struct PrintStmt : Stmt {
    std::shared_ptr<Expr> expr;
    PrintStmt(std::shared_ptr<Expr> e) : expr(e) {}
    void execute(Environment& env) override {
        std::cout << expr->evaluate(env).toString() << std::endl;
    }
};

struct LetStmt : Stmt {
    std::string name;
    std::shared_ptr<Expr> init;
    LetStmt(std::string n, std::shared_ptr<Expr> i) : name(n), init(i) {}
    void execute(Environment& env) override {
        env.define(name, init->evaluate(env));
    }
};

struct AssignStmt : Stmt {
    std::string name;
    std::shared_ptr<Expr> value;
    AssignStmt(std::string n, std::shared_ptr<Expr> v) : name(n), value(v) {}
    void execute(Environment& env) override {
        env.assign(name, value->evaluate(env));
    }
};

struct IfStmt : Stmt {
    std::shared_ptr<Expr> condition;
    std::shared_ptr<Stmt> thenBranch;
    std::shared_ptr<Stmt> elseBranch;
    IfStmt(std::shared_ptr<Expr> c, std::shared_ptr<Stmt> t, std::shared_ptr<Stmt> e)
        : condition(c), thenBranch(t), elseBranch(e) {}

    void execute(Environment& env) override {
        if (condition->evaluate(env).isTruthy()) {
            thenBranch->execute(env);
        } else if (elseBranch) {
            elseBranch->execute(env);
        }
    }
};

struct WhileStmt : Stmt {
    std::shared_ptr<Expr> condition;
    std::shared_ptr<Stmt> body;
    WhileStmt(std::shared_ptr<Expr> c, std::shared_ptr<Stmt> b) : condition(c), body(b) {}

    void execute(Environment& env) override {
        while (condition->evaluate(env).isTruthy()) {
            Environment loopEnv(&env);
            body->execute(loopEnv);
        }
    }
};

struct ForStmt : Stmt {
    std::string varName;
    int start, end;
    std::shared_ptr<Stmt> body;
    ForStmt(std::string v, int s, int e, std::shared_ptr<Stmt> b)
        : varName(v), start(s), end(e), body(b) {}

    void execute(Environment& env) override {
        Environment loopEnv(&env);
        for (int i = start; i <= end; ++i) {
            loopEnv.define(varName, Value(i));
            body->execute(loopEnv);
        }
    }
};

// --- 3. Parser ---

class Parser {
    std::vector<Token> tokens;
    size_t current = 0;

public:
    Parser(const std::vector<Token>& t) : tokens(t) {}

    std::shared_ptr<NestStmt> parseProgram() {
        consume(KEYWORD_NEST, "Expect 'nest' at start of program.");
        consume(IDENTIFIER, "Expect program name after 'nest'.");
        std::string name = previous().text;
        if (name != "main") throw std::runtime_error("Entry point must be 'nest main'");

        consume(LBRACE, "Expect '{' before nest body.");
        auto body = parseBlock();
        return std::make_shared<NestStmt>(name, body);
    }

private:
    std::shared_ptr<BlockStmt> parseBlock() {
        auto block = std::make_shared<BlockStmt>();
        while (!check(RBRACE) && !isAtEnd()) {
            block->statements.push_back(parseStatement());
        }
        consume(RBRACE, "Expect '}' after block.");
        return block;
    }

    std::shared_ptr<Stmt> parseStatement() {
        if (match(KEYWORD_NEST)) return parseNest();
        if (match(KEYWORD_LET)) return parseLet();
        if (match(KEYWORD_PRINT)) return parsePrint();
        if (match(KEYWORD_IF)) return parseIf();
        if (match(KEYWORD_WHILE)) return parseWhile();
        if (match(KEYWORD_FOR)) return parseFor();

        if (check(IDENTIFIER)) {
            Token nextTok = tokens[current + 1];
            if (nextTok.type == ASSIGN) return parseAssign();
        }

        throw std::runtime_error("Expect statement at line " + std::to_string(peek().line));
    }

    std::shared_ptr<Stmt> parseNest() {
        consume(IDENTIFIER, "Expect nest name.");
        std::string name = previous().text;
        consume(LBRACE, "Expect '{' after nest name.");
        return std::make_shared<NestStmt>(name, parseBlock());
    }

    std::shared_ptr<Stmt> parseLet() {
        consume(IDENTIFIER, "Expect variable name.");
        std::string name = previous().text;
        consume(ASSIGN, "Expect '=' after variable name.");
        auto expr = parseExpression();
        return std::make_shared<LetStmt>(name, expr);
    }

    std::shared_ptr<Stmt> parseAssign() {
        consume(IDENTIFIER, "Expect variable name.");
        std::string name = previous().text;
        consume(ASSIGN, "Expect '='.");
        auto expr = parseExpression();
        return std::make_shared<AssignStmt>(name, expr);
    }

    std::shared_ptr<Stmt> parsePrint() {
        consume(LPAREN, "Expect '(' after print.");
        auto expr = parseExpression();
        consume(RPAREN, "Expect ')' after value.");
        return std::make_shared<PrintStmt>(expr);
    }

    std::shared_ptr<Stmt> parseIf() {
        consume(LPAREN, "Expect '(' after if.");
        auto condition = parseExpression();
        consume(RPAREN, "Expect ')' after if condition.");
        consume(LBRACE, "Expect '{' before if body.");
        auto thenBranch = parseBlock();
        std::shared_ptr<Stmt> elseBranch = nullptr;
        if (match(KEYWORD_ELSE)) {
            consume(LBRACE, "Expect '{' before else body.");
            elseBranch = parseBlock();
        }
        return std::make_shared<IfStmt>(condition, thenBranch, elseBranch);
    }

    std::shared_ptr<Stmt> parseWhile() {
        consume(LPAREN, "Expect '(' after while.");
        auto condition = parseExpression();
        consume(RPAREN, "Expect ')' after condition.");
        consume(LBRACE, "Expect '{' before while body.");
        auto body = parseBlock();
        return std::make_shared<WhileStmt>(condition, body);
    }

    std::shared_ptr<Stmt> parseFor() {
        consume(LPAREN, "Expect '(' after for.");
        consume(IDENTIFIER, "Expect variable name in for.");
        std::string var = previous().text;
        consume(KEYWORD_IN, "Expect 'in' after variable.");

        consume(NUMBER, "Expect start number.");
        int start = std::stoi(previous().text);
        consume(DOTDOT, "Expect '..' range operator.");
        consume(NUMBER, "Expect end number.");
        int end = std::stoi(previous().text);

        consume(RPAREN, "Expect ')' after for clause.");
        consume(LBRACE, "Expect '{' before for body.");
        auto body = parseBlock();
        return std::make_shared<ForStmt>(var, start, end, body);
    }

    std::shared_ptr<Expr> parseExpression() { return parseEquality(); }

    std::shared_ptr<Expr> parseEquality() {
        auto expr = parseComparison();
        while (match(EQ, NEQ)) {
            TokenType op = previous().type;
            auto right = parseComparison();
            expr = std::make_shared<BinaryExpr>(expr, op, right);
        }
        return expr;
    }

    std::shared_ptr<Expr> parseComparison() {
        auto expr = parseTerm();
        while (match(GT, GTE, LT, LTE)) {
            TokenType op = previous().type;
            auto right = parseTerm();
            expr = std::make_shared<BinaryExpr>(expr, op, right);
        }
        return expr;
    }

    std::shared_ptr<Expr> parseTerm() {
        auto expr = parseFactor();
        while (match(PLUS, MINUS)) {
            TokenType op = previous().type;
            auto right = parseFactor();
            expr = std::make_shared<BinaryExpr>(expr, op, right);
        }
        return expr;
    }

    std::shared_ptr<Expr> parseFactor() {
        auto expr = parsePostfix();
        while (match(STAR, SLASH, PERCENT)) {
            TokenType op = previous().type;
            auto right = parsePostfix();
            expr = std::make_shared<BinaryExpr>(expr, op, right);
        }
        return expr;
    }

    std::shared_ptr<Expr> parsePostfix() {
        auto expr = parsePrimary();
        while (true) {
            if (match(LBRACKET)) {
                auto index = parseExpression();
                consume(RBRACKET, "Expect ']' after index.");
                expr = std::make_shared<IndexExpr>(expr, index);
            } else {
                break;
            }
        }
        return expr;
    }

    std::shared_ptr<Expr> parsePrimary() {
        if (match(NUMBER)) return std::make_shared<LiteralExpr>(Value(std::stoi(previous().text)));
        if (match(STRING_LITERAL)) return std::make_shared<LiteralExpr>(Value(previous().text));
        if (match(IDENTIFIER)) return std::make_shared<VariableExpr>(previous().text);
        if (match(LPAREN)) {
            auto expr = parseExpression();
            consume(RPAREN, "Expect ')' after expression.");
            return expr;
        }
        if (match(LBRACKET)) return parseArray();
        if (match(LBRACE)) return parseDict();
        throw std::runtime_error("Expect expression at line " + std::to_string(peek().line));
    }

    std::shared_ptr<Expr> parseArray() {
        std::vector<std::shared_ptr<Expr>> elements;
        if (!check(RBRACKET)) {
            do {
                elements.push_back(parseExpression());
            } while (match(COMMA));
        }
        consume(RBRACKET, "Expect ']' after array elements.");
        return std::make_shared<ArrayExpr>(elements);
    }

    std::shared_ptr<Expr> parseDict() {
        std::map<std::string, std::shared_ptr<Expr>> pairs;
        if (!check(RBRACE)) {
            do {
                consume(STRING_LITERAL, "Expect string key in dict.");
                std::string key = previous().text;
                consume(COLON, "Expect ':' after dict key.");
                auto value = parseExpression();
                pairs[key] = value;
            } while (match(COMMA));
        }
        consume(RBRACE, "Expect '}' after dict pairs.");
        return std::make_shared<DictExpr>(pairs);
    }

    // Helpers - using basic parameter pack expansion for C++11/14 compatibility check if needed,
    // but simplifying to explicit overloading or list to be safe.
    // Re-implementing match to be simple and safe for C++11
    bool match(TokenType t1) {
        if (check(t1)) { advance(); return true; }
        return false;
    }
    bool match(TokenType t1, TokenType t2) {
        if (check(t1) || check(t2)) { advance(); return true; }
        return false;
    }
    bool match(TokenType t1, TokenType t2, TokenType t3) {
        if (check(t1) || check(t2) || check(t3)) { advance(); return true; }
        return false;
    }
    bool match(TokenType t1, TokenType t2, TokenType t3, TokenType t4) {
        if (check(t1) || check(t2) || check(t3) || check(t4)) { advance(); return true; }
        return false;
    }

    bool check(TokenType type) {
        if (isAtEnd()) return false;
        return peek().type == type;
    }

    Token advance() {
        if (!isAtEnd()) current++;
        return previous();
    }

    bool isAtEnd() { return peek().type == TOKEN_EOF; }
    Token peek() { return tokens[current]; }
    Token previous() { return tokens[current - 1]; }

    Token consume(TokenType type, std::string message) {
        if (check(type)) return advance();
        throw std::runtime_error(message + " Got: " + peek().text);
    }
};

// --- Main ---

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: nest <script.nest>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    try {
        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        Parser parser(tokens);
        auto program = parser.parseProgram();

        Environment globalEnv;
        program->execute(globalEnv);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}