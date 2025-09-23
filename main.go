package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/big"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type Expr struct {
	Val  *big.Rat
	Repr string
}

type interval struct{ l, r int }

// op symbols
const (
	opAdd = "+"
	opSub = "-"
	opMul = "*"
	opDiv = "/"
	opPow = "^"
)

type config struct {
	order     string
	limit     int
	workers   int
	maxExp    int
	targetStr string
	target    *big.Rat
}

func parseRat(s string) (*big.Rat, error) {
	r := new(big.Rat)
	if _, ok := r.SetString(s); ok {
		return r, nil
	}
	return nil, fmt.Errorf("cannot parse number: %s", s)
}

func makeDigits(order string) []int {
	if order == "desc" {
		return []int{9, 8, 7, 6, 5, 4, 3, 2, 1}
	}
	return []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
}

func combineMaps(left, right map[string]Expr, maxExp int) map[string]Expr {
	out := make(map[string]Expr, len(left)*len(right))
	for _, L := range left {
		for _, R := range right {
			// +
			{
				v := new(big.Rat).Add(L.Val, R.Val)
				addIfNew(out, v, fmt.Sprintf("(%s%s%s)", L.Repr, opAdd, R.Repr))
			}
			// -
			{
				v := new(big.Rat).Sub(L.Val, R.Val)
				addIfNew(out, v, fmt.Sprintf("(%s%s%s)", L.Repr, opSub, R.Repr))
			}
			// *
			{
				v := new(big.Rat).Mul(L.Val, R.Val)
				addIfNew(out, v, fmt.Sprintf("(%s%s%s)", L.Repr, opMul, R.Repr))
			}
			// /
			if R.Val.Sign() != 0 {
				v := new(big.Rat).Quo(L.Val, R.Val)
				addIfNew(out, v, fmt.Sprintf("(%s%s%s)", L.Repr, opDiv, R.Repr))
			}
			// ^
			if isInteger(R.Val) {
				exp := int(R.Val.Num().Int64())
				if abs(exp) <= maxExp {
					if powVal, ok := powRat(L.Val, exp); ok {
						addIfNew(out, powVal, fmt.Sprintf("(%s%s%s)", L.Repr, opPow, R.Repr))
					}
				}
			}
		}
	}
	return out
}

func addIfNew(m map[string]Expr, val *big.Rat, repr string) {
	key := ratKey(val)
	if _, exists := m[key]; !exists {
		m[key] = Expr{Val: new(big.Rat).Set(val), Repr: repr}
	}
}

func ratKey(r *big.Rat) string  { return r.RatString() }
func isInteger(r *big.Rat) bool { return r.IsInt() }
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// 整数指数の累乗（負の指数は 1/x^|n|）
func powRat(base *big.Rat, exp int) (*big.Rat, bool) {
	if exp == 0 {
		return big.NewRat(1, 1), true
	}
	a := new(big.Int).Set(base.Num())
	b := new(big.Int).Set(base.Denom())
	if exp < 0 {
		a, b = b, a
		exp = -exp
		if b.Sign() == 0 {
			return nil, false
		}
	}
	anum := new(big.Int).SetInt64(1)
	aden := new(big.Int).SetInt64(1)
	bn := new(big.Int).Set(a)
	bd := new(big.Int).Set(b)
	e := exp
	for e > 0 {
		if e&1 == 1 {
			anum.Mul(anum, bn)
			aden.Mul(aden, bd)
		}
		bn.Mul(bn, bn)
		bd.Mul(bd, bd)
		e >>= 1
	}
	res := new(big.Rat).SetFrac(anum, aden) // 既約化される
	return res, true
}

// digits[i..j] をそのまま連結した整数（例：1,2,3 → 123）
func concatRat(digits []int, i, j int) (*big.Rat, string) {
	var sb strings.Builder
	val := big.NewInt(0)
	ten := big.NewInt(10)
	for k := i; k <= j; k++ {
		sb.WriteString(strconv.Itoa(digits[k]))
		val.Mul(val, ten)
		val.Add(val, big.NewInt(int64(digits[k])))
	}
	return new(big.Rat).SetInt(val), sb.String()
}

func main() {
	cfg := config{}
	defaultWorkers := runtime.NumCPU()
	flag.StringVar(&cfg.targetStr, "target", "", "目標値（整数・分数 22/7・小数 3.5 など）")
	flag.StringVar(&cfg.order, "order", "asc", "数字の順序: asc（1→9）/ desc（9→1）")
	flag.IntVar(&cfg.limit, "limit", 10, "表示する最大件数")
	flag.IntVar(&cfg.workers, "workers", defaultWorkers, "ワーカー数（並列度）")
	flag.IntVar(&cfg.maxExp, "maxexp", 5, "指数の絶対値上限（^ の右辺が整数のときのみ適用）")
	flag.Parse()

	if cfg.targetStr == "" {
		log.Fatal("usage: -target <number> （例: -target 100）")
	}
	var err error
	cfg.target, err = parseRat(cfg.targetStr)
	if err != nil {
		log.Fatalf("target parse error: %v", err)
	}
	if cfg.workers < 1 {
		cfg.workers = 1
	}
	if cfg.order != "asc" && cfg.order != "desc" {
		log.Fatalf("order must be asc or desc")
	}

	digits := makeDigits(cfg.order)
	n := len(digits)

	// DP: cache[interval] -> map[valueKey]Expr
	cache := make(map[interval]map[string]Expr)

	// まず全区間について「連結した数」を初期候補として入れておく
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			key := interval{i, j}
			r, repr := concatRat(digits, i, j)
			if cache[key] == nil {
				cache[key] = make(map[string]Expr)
			}
			cache[key][ratKey(r)] = Expr{Val: r, Repr: repr}
		}
	}

	// worker pool
	type job struct {
		l, r  int
		left  map[string]Expr
		right map[string]Expr
	}
	type result struct {
		l, r int
		data map[string]Expr
	}

	jobCh := make(chan job, 1024)
	resCh := make(chan result, 1024)
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for j := range jobCh {
			data := combineMaps(j.left, j.right, cfg.maxExp)
			resCh <- result{l: j.l, r: j.r, data: data}
		}
	}
	wg.Add(cfg.workers)
	for i := 0; i < cfg.workers; i++ {
		go worker()
	}

	// 区間DP（全ての括弧付け）: 連結初期値に加えて演算で作れる値を統合
	for length := 2; length <= n; length++ {
		intervals := make([]interval, 0, n-length+1)
		for l := 0; l+length-1 < n; l++ {
			r := l + length - 1
			intervals = append(intervals, interval{l, r})
		}

		// 分割をジョブ化
		type expect struct{ l, r, count int }
		expCounts := []expect{}
		for _, iv := range intervals {
			cnt := 0
			for k := iv.l; k < iv.r; k++ {
				left := cache[interval{iv.l, k}]
				right := cache[interval{k + 1, iv.r}]
				if len(left) == 0 || len(right) == 0 {
					continue
				}
				cnt++
				jobCh <- job{
					l:     iv.l,
					r:     iv.r,
					left:  left,
					right: right,
				}
			}
			expCounts = append(expCounts, expect{iv.l, iv.r, cnt})
		}

		// 結果をマージ
		done := make(chan struct{})
		var mu sync.Mutex
		go func() {
			remaining := 0
			for _, e := range expCounts {
				remaining += e.count
			}
			for remaining > 0 {
				res := <-resCh
				mu.Lock()
				if cache[interval{res.l, res.r}] == nil {
					cache[interval{res.l, res.r}] = make(map[string]Expr)
				}
				for k, v := range res.data {
					if _, exists := cache[interval{res.l, res.r}][k]; !exists {
						cache[interval{res.l, res.r}][k] = v
					}
				}
				mu.Unlock()
				remaining--
			}
			close(done)
		}()
		<-done
	}

	close(jobCh)
	wg.Wait()
	close(resCh)

	// 全区間
	full := cache[interval{0, n - 1}]
	if len(full) == 0 {
		fmt.Println("式が見つかりませんでした。")
		return
	}

	// 目標値一致を収集
	targetKey := ratKey(cfg.target)
	candidates := make([]string, 0, 32)
	for k, e := range full {
		if k == targetKey {
			candidates = append(candidates, e.Repr)
		}
	}

	if len(candidates) == 0 {
		type near struct {
			diff float64
			expr string
			val  *big.Rat
		}
		nears := make([]near, 0, 50)
		for _, e := range full {
			d := absFloat(new(big.Rat).Sub(e.Val, cfg.target))
			nears = append(nears, near{diff: d, expr: e.Repr, val: e.Val})
		}
		sort.Slice(nears, func(i, j int) bool { return nears[i].diff < nears[j].diff })
		fmt.Printf("完全一致の式は見つかりませんでした。近い式をいくつか表示します（上位 %d 件）：\n", cfg.limit)
		for i := 0; i < len(nears) && i < cfg.limit; i++ {
			fmt.Printf("[%2d] %s = %s  (誤差 %.6g)\n", i+1, nears[i].expr, nears[i].val.RatString(), nears[i].diff)
		}
		return
	}

	// 重複除去＆表示
	candidates = uniqueStrings(candidates)
	fmt.Printf("目標 %s に一致した式（最大 %d 件）:\n", cfg.targetStr, cfg.limit)
	for i := 0; i < len(candidates) && i < cfg.limit; i++ {
		fmt.Printf("[%2d] %s = %s\n", i+1, candidates[i], cfg.targetStr)
	}
}

func absFloat(r *big.Rat) float64 {
	f, _ := r.Float64()
	if f < 0 {
		f = -f
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return math.MaxFloat64
	}
	return f
}

func uniqueStrings(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}
