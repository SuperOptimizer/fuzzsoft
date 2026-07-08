//! fs-covmap: offline coverage-attribution tool.
//!
//! Consumes a `System.map` (nm-format symbol table) and an exact edge dump
//! produced by `fuzzsoft run <elf> --cov-out FILE` (or a plain list of hex
//! PCs, one per line), resolves each covered PC to the kernel text symbol
//! (function) that contains it, and reports:
//!
//!   1. distinct PCs seen / resolved / unresolved
//!   2. distinct kernel functions covered
//!   3. top-N covered functions by distinct-PC count
//!   4. subsystem attribution: covered/total text symbols per named
//!      subsystem, to directly show which subsystems (e.g. io_uring, bpf)
//!      are structurally unreached by the current fuzzing corpus/harness.
//!
//! Zero dependencies (std only), hand-rolled arg parsing to match the rest
//! of the fuzzsoft repo (see crates/fs-cli).
//!
//! Usage: `fs-covmap <System.map> <cov-dump> [--top N] [--json]`

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::process::ExitCode;

/// A text (code) symbol from System.map: address, name, and size (computed
/// as distance to the next text-or-any symbol, or `None` if it's the last
/// symbol in the map / size can't be determined).
#[derive(Debug, Clone)]
struct Symbol {
    addr: u32,
    name: String,
}

/// Sorted-by-address table of text symbols, supporting address -> symbol
/// containment lookup.
struct SymTable {
    syms: Vec<Symbol>,
}

impl SymTable {
    /// Parse a System.map file's contents. Keeps only `[tT]`-typed (text)
    /// symbols, sorted ascending by address (System.map is normally already
    /// sorted, but we sort defensively).
    fn parse(text: &str) -> SymTable {
        let mut syms = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let (Some(addr_s), Some(ty_s), Some(name_s)) = (it.next(), it.next(), it.next())
            else {
                continue;
            };
            if !matches!(ty_s, "t" | "T") {
                continue;
            }
            let Ok(addr) = u32::from_str_radix(addr_s, 16) else {
                continue;
            };
            syms.push(Symbol {
                addr,
                name: name_s.to_string(),
            });
        }
        syms.sort_by_key(|s| s.addr);
        syms.dedup_by_key(|s| s.addr);
        SymTable { syms }
    }

    /// Resolve a PC to the text symbol whose address is the greatest one
    /// `<= pc`, provided `pc` is also `<` the next text symbol's address
    /// (i.e. still "inside" that symbol's range). Returns `None` if `pc` is
    /// before the first text symbol, or `>=` the last text symbol's
    /// address plus its implied range (we treat the last symbol as having
    /// no known upper bound... but per spec we resolve strictly "past the
    /// end / unknown" -> None only when there IS a following symbol and pc
    /// has run past it; for a PC at or after the very last symbol's
    /// address, there's no "next" boundary to violate, so per the spec's
    /// alternative ("or last-symbol per your rule") we choose: the last
    /// symbol extends to infinity is NOT what we want for "past the last
    /// resolves to None" case in the doc -- we pick None for a PC clearly
    /// past the final symbol by more than a sane bound is ambiguous, so we
    /// simply say: PC >= last symbol's address resolves to the last
    /// symbol (open-ended range), UNLESS pc < first symbol addr -> None.
    ///
    /// Concretely: binary search for the last symbol with addr <= pc.
    /// If none exists (pc < syms[0].addr), return None. Otherwise return
    /// that symbol. This means the very last symbol's range is open-ended
    /// (extends to the end of the address space), matching "past the last
    /// resolves to ... last-symbol per your rule" -- see unit test
    /// `resolve_past_last_is_last_symbol` for the chosen behavior, and
    /// `resolve_before_first_is_none` for the other edge.
    fn resolve(&self, pc: u32) -> Option<&Symbol> {
        if self.syms.is_empty() {
            return None;
        }
        // Find the partition point: first index where syms[i].addr > pc.
        let idx = self.syms.partition_point(|s| s.addr <= pc);
        if idx == 0 {
            return None;
        }
        Some(&self.syms[idx - 1])
    }
}

/// Parse a coverage dump: either the exact edge-dump format emitted by
/// `fuzzsoft run --cov-out` (`# comment` header line, then
/// `<from_hex> <to_hex>` per line), or a plain list of one hex PC per line.
/// Returns the set of distinct PCs touched. By default only `to` targets
/// count (the block-entry PC that identifies reached code); `from` PCs are
/// also folded in via `--edges` mode... but per the spec, default is `to`
/// only for edge lines, while single-PC lines are taken as-is.
fn parse_cov_dump(text: &str, edges_mode: bool) -> Vec<u32> {
    let mut pcs = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.as_slice() {
            [from, to] => {
                if edges_mode {
                    if let Some(v) = parse_hex_pc(from) {
                        pcs.push(v);
                    }
                }
                if let Some(v) = parse_hex_pc(to) {
                    pcs.push(v);
                }
            }
            [only] => {
                if let Some(v) = parse_hex_pc(only) {
                    pcs.push(v);
                }
            }
            _ => {}
        }
    }
    pcs
}

fn parse_hex_pc(tok: &str) -> Option<u32> {
    let tok = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X"))?;
    u32::from_str_radix(tok, 16).ok()
}

/// Named subsystem buckets for symbol-name attribution. First match (in
/// this order) wins; anything left over falls into "other".
///
/// NOTE: `ep_` (eventpoll) is intentionally handled as a *prefix* check in
/// `bucket_of`, not a plain substring like the others -- as a substring it
/// matches all over the kernel (`prep_pwqs`, `ptep_set_access_flags`,
/// `sleep_fs_sync`, ...) since "ep_" is a common English/C fragment. The
/// spec text calls this out explicitly ("`ep_` prefix").
const SUBSYSTEMS: &[(&str, &[&str])] = &[
    ("io_uring", &["io_uring"]),
    ("bpf", &["bpf_", "bpf"]),
    ("eventpoll", &["eventpoll", "epoll"]),
    ("keyctl", &["keyctl", "key_", "keyring"]),
    (
        "mount/fs",
        &["do_mount", "path_mount", "mount_", "vfs_", "overlayfs", "ovl_"],
    ),
    (
        "splice",
        &["splice", "vmsplice", "tee_", "iter_file_splice"],
    ),
    (
        "namespaces",
        &[
            "unshare",
            "setns",
            "copy_namespaces",
            "create_new_namespaces",
            "nsproxy",
        ],
    ),
    (
        "slab/alloc",
        &[
            "kmalloc",
            "kfree",
            "__kmalloc",
            "slab",
            "kmem_cache",
            "alloc_pages",
            "__alloc",
        ],
    ),
];

/// The known-gap subsystems the doc specifically calls out; these get a
/// verdict line in the report.
const KNOWN_GAP_SUBSYSTEMS: &[&str] = &[
    "io_uring",
    "bpf",
    "keyctl",
    "mount/fs",
    "splice",
    "namespaces",
];

fn bucket_of(name: &str) -> &'static str {
    for (bucket, needles) in SUBSYSTEMS {
        // `ep_` is a prefix check (see SUBSYSTEMS doc comment): as a
        // substring it would false-positive on unrelated symbols like
        // `prep_pwqs` or `ptep_set_access_flags`.
        if *bucket == "eventpoll" && name.starts_with("ep_") {
            return bucket;
        }
        for needle in *needles {
            if name.contains(needle) {
                return bucket;
            }
        }
    }
    "other"
}

struct Args {
    map_path: String,
    cov_path: String,
    top: usize,
    json: bool,
    edges_mode: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut top = 25usize;
    let mut json = false;
    let mut edges_mode = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--top" => {
                let v = it.next().ok_or("--top needs a value")?;
                top = v.parse::<usize>().map_err(|_| "--top needs an integer")?;
            }
            "--json" => json = true,
            "--edges" => edges_mode = true,
            "-h" | "--help" => return Err("help".to_string()),
            other if other.starts_with("--") => {
                return Err(format!("unknown flag: {other}"));
            }
            other => positional.push(other.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err("expected <System.map> <cov-dump>".to_string());
    }
    Ok(Args {
        map_path: positional[0].clone(),
        cov_path: positional[1].clone(),
        top,
        json,
        edges_mode,
    })
}

fn usage() -> String {
    "usage: fs-covmap <System.map> <cov-dump> [--top N] [--json] [--edges]\n\
     \n\
     <System.map>  nm-format kernel symbol table (sorted by address)\n\
     <cov-dump>    edge dump from `fuzzsoft run <elf> --cov-out FILE`,\n\
                   or a plain list of one hex PC per line\n\
     --top N       show top N covered symbols by distinct-PC count (default 25)\n\
     --json        emit a machine-readable subsystem summary instead of prose\n\
     --edges       also count `from` PCs (default: only `to` targets count)\n"
        .to_string()
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw_args) {
        Ok(a) => a,
        Err(e) => {
            if e == "help" {
                print!("{}", usage());
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {e}");
            eprint!("{}", usage());
            return ExitCode::FAILURE;
        }
    };

    let map_text = match std::fs::read_to_string(&args.map_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: reading {}: {e}", args.map_path);
            return ExitCode::FAILURE;
        }
    };
    let cov_text = match std::fs::read_to_string(&args.cov_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: reading {}: {e}", args.cov_path);
            return ExitCode::FAILURE;
        }
    };

    let table = SymTable::parse(&map_text);
    let mut pcs = parse_cov_dump(&cov_text, args.edges_mode);
    pcs.sort_unstable();
    pcs.dedup();

    // Footgun guard: a dump with real content but zero parsed PCs almost always
    // means a format mismatch (bare hex vs the `0x`-prefixed form `fuzzsoft run
    // --cov-out` emits). Silently reporting an all-zero "total gap" would be a lie
    // for a progress gauge, so warn loudly instead.
    if pcs.is_empty() {
        if let Some(sample) = cov_text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with('#'))
        {
            eprintln!(
                "warning: parsed 0 PCs from a non-empty dump — expected 0x-prefixed hex \
                 (e.g. `0xc00185e0 0xc00185e0`); first content line was: {sample:?}"
            );
        }
    }

    // Total text symbols per subsystem bucket (from the full System.map).
    let mut total_per_bucket: BTreeMap<&'static str, usize> = BTreeMap::new();
    for s in &table.syms {
        *total_per_bucket.entry(bucket_of(&s.name)).or_insert(0) += 1;
    }

    // Resolve each covered PC; tally distinct PCs per symbol.
    let mut pcs_per_symbol: BTreeMap<String, usize> = BTreeMap::new();
    let mut unresolved = 0usize;
    for &pc in &pcs {
        match table.resolve(pc) {
            Some(sym) => *pcs_per_symbol.entry(sym.name.clone()).or_insert(0) += 1,
            None => unresolved += 1,
        }
    }
    let resolved = pcs.len() - unresolved;

    // Covered symbols per subsystem bucket.
    let mut covered_per_bucket: BTreeMap<&'static str, usize> = BTreeMap::new();
    for name in pcs_per_symbol.keys() {
        *covered_per_bucket.entry(bucket_of(name)).or_insert(0) += 1;
    }

    if args.json {
        print_json(&total_per_bucket, &covered_per_bucket);
        return ExitCode::SUCCESS;
    }

    println!("fs-covmap: coverage attribution");
    println!("  System.map        : {}", args.map_path);
    println!("  cov-dump          : {}", args.cov_path);
    println!("  text symbols total: {}", table.syms.len());
    println!();
    println!("distinct PCs in dump : {}", pcs.len());
    println!("  resolved to symbol : {resolved}");
    println!("  unresolved         : {unresolved}");
    println!("distinct functions covered: {}", pcs_per_symbol.len());
    println!();

    println!("top {} covered symbols (by distinct PCs hit):", args.top);
    let mut ranked: Vec<(&String, &usize)> = pcs_per_symbol.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (name, count) in ranked.iter().take(args.top) {
        println!("  {count:6}  {name}");
    }
    println!();

    println!("subsystem attribution (covered symbols / total symbols in System.map):");
    let mut all_buckets: Vec<&'static str> =
        SUBSYSTEMS.iter().map(|(b, _)| *b).collect();
    all_buckets.push("other");
    for bucket in &all_buckets {
        let total = *total_per_bucket.get(bucket).unwrap_or(&0);
        let covered = *covered_per_bucket.get(bucket).unwrap_or(&0);
        println!("  {bucket:<12} {covered:5} / {total:<5}");
    }
    println!();

    println!("known-gap verdicts:");
    for bucket in KNOWN_GAP_SUBSYSTEMS {
        let total = *total_per_bucket.get(bucket).unwrap_or(&0);
        let covered = *covered_per_bucket.get(bucket).unwrap_or(&0);
        if covered == 0 {
            println!("  {bucket:<12} GAP CONFIRMED (0/{total} covered)");
        } else {
            println!("  {bucket:<12} partial ({covered}/{total})");
        }
    }

    ExitCode::SUCCESS
}

fn print_json(
    total_per_bucket: &BTreeMap<&'static str, usize>,
    covered_per_bucket: &BTreeMap<&'static str, usize>,
) {
    let mut all_buckets: Vec<&'static str> = SUBSYSTEMS.iter().map(|(b, _)| *b).collect();
    all_buckets.push("other");
    let mut out = String::new();
    let _ = write!(out, "{{");
    for (i, bucket) in all_buckets.iter().enumerate() {
        if i > 0 {
            let _ = write!(out, ",");
        }
        let total = *total_per_bucket.get(bucket).unwrap_or(&0);
        let covered = *covered_per_bucket.get(bucket).unwrap_or(&0);
        let _ = write!(
            out,
            "\"{bucket}\":{{\"total\":{total},\"covered\":{covered}}}"
        );
    }
    let _ = write!(out, "}}");
    println!("{out}");
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_MAP: &str = "\
c0000000 T _start
c0000044 t coff_header
c0000100 T do_mount
c0000200 t some_local_fn
";

    #[test]
    fn resolve_middle_of_start_is_start() {
        let table = SymTable::parse(TINY_MAP);
        // _start spans [c0000000, c0000044); a PC in the middle resolves to _start.
        let sym = table.resolve(0xc0000020).expect("should resolve");
        assert_eq!(sym.name, "_start");
    }

    #[test]
    fn resolve_before_first_is_none() {
        let table = SymTable::parse(TINY_MAP);
        assert!(table.resolve(0xbfffffff).is_none());
        assert!(table.resolve(0).is_none());
    }

    #[test]
    fn resolve_past_last_is_last_symbol() {
        let table = SymTable::parse(TINY_MAP);
        // Past the final symbol's address: open-ended range, resolves to
        // the last symbol (some_local_fn) rather than None, per our chosen
        // containment rule (see SymTable::resolve doc comment).
        let sym = table.resolve(0xc0000fff).expect("resolves to last symbol");
        assert_eq!(sym.name, "some_local_fn");
    }

    #[test]
    fn resolve_exact_symbol_address() {
        let table = SymTable::parse(TINY_MAP);
        let sym = table.resolve(0xc0000100).expect("exact address resolves");
        assert_eq!(sym.name, "do_mount");
    }

    #[test]
    fn ignores_non_text_symbols() {
        let map = "\
c0000000 T _start
c0000010 D some_data
c0000020 t after_data
";
        let table = SymTable::parse(map);
        assert_eq!(table.syms.len(), 2);
        // A PC that would land "inside" the data symbol's range (if it were
        // counted) still resolves to _start, since D is ignored.
        let sym = table.resolve(0xc0000015).expect("resolves");
        assert_eq!(sym.name, "_start");
    }

    #[test]
    fn bucket_io_uring() {
        assert_eq!(bucket_of("io_uring_submit"), "io_uring");
    }

    #[test]
    fn bucket_bpf() {
        assert_eq!(bucket_of("bpf_prog_run"), "bpf");
    }

    #[test]
    fn bucket_eventpoll() {
        assert_eq!(bucket_of("ep_loop_check_proc"), "eventpoll");
        assert_eq!(bucket_of("eventpoll_release"), "eventpoll");
    }

    #[test]
    fn bucket_other_fallback() {
        assert_eq!(bucket_of("totally_unrelated_fn"), "other");
    }

    #[test]
    fn bucket_precedence_bpf_before_other_but_after_io_uring() {
        // io_uring is checked before bpf; a name matching both should hit
        // io_uring first per listed order (constructed edge case).
        assert_eq!(bucket_of("io_uring_bpf_helper"), "io_uring");
    }

    #[test]
    fn parse_edge_dump_default_counts_to_only() {
        let dump = "# fuzzsoft coverage: 2 edges\n0xc0000000 0xc0000044\n0xc0000044 0xc0000100\n";
        let mut pcs = parse_cov_dump(dump, false);
        pcs.sort_unstable();
        pcs.dedup();
        assert_eq!(pcs, vec![0xc0000044, 0xc0000100]);
    }

    #[test]
    fn parse_edge_dump_edges_mode_counts_both() {
        let dump = "# fuzzsoft coverage: 2 edges\n0xc0000000 0xc0000044\n0xc0000044 0xc0000100\n";
        let mut pcs = parse_cov_dump(dump, true);
        pcs.sort_unstable();
        pcs.dedup();
        assert_eq!(pcs, vec![0xc0000000, 0xc0000044, 0xc0000100]);
    }

    #[test]
    fn parse_plain_pc_list() {
        let dump = "0xc0000000\n0xc0000100\n0xc0000100\n";
        let mut pcs = parse_cov_dump(dump, false);
        pcs.sort_unstable();
        pcs.dedup();
        assert_eq!(pcs, vec![0xc0000000, 0xc0000100]);
    }
}
