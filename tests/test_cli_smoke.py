from pathlib import Path
from tools import evaluate, export_report, prune_and_render, visualize_bad_gaussians

def test_cli_mains_run(tmp_path: Path, monkeypatch) -> None:
    eval_dir=tmp_path/"eval"
    monkeypatch.setattr("sys.argv", ["evaluate","--scene","dummy.npz","--cameras","dummy.json","--targets","dummy.npy","--baseline","hybrid","--report-dir",str(eval_dir)])
    evaluate.main()
    summary=eval_dir/"summary.json"
    rep=tmp_path/"report"
    monkeypatch.setattr("sys.argv", ["export_report","--input-json",str(summary),"--report-dir",str(rep)])
    export_report.main()
    prune_out=tmp_path/"pruned.json"
    monkeypatch.setattr("sys.argv", ["prune","--scene","dummy.npz","--confidence",str(summary),"--output",str(prune_out)])
    prune_and_render.main()
    vis_out=tmp_path/"bad.json"
    monkeypatch.setattr("sys.argv", ["vis","--scene","dummy.npz","--output",str(vis_out),"--top-k","5"])
    visualize_bad_gaussians.main()
    assert summary.exists() and (rep/"report.html").exists() and prune_out.exists() and vis_out.exists()
