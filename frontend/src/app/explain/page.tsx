export default function ExplainPage() {
  return (
    <div className="min-h-screen bg-base-950 text-zinc-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-semibold mb-4">Explainability</h1>
        <p className="text-sm text-zinc-400 mb-6">
          This page will present SHAP explanations, feature attributions, and root-cause details.
        </p>
        <div className="rounded-3xl border border-base-700 bg-base-900 p-8 shadow-xl shadow-black/20">
          <p className="text-zinc-300 leading-relaxed">
            If you see this page, the route is now returning a valid React component.
          </p>
        </div>
      </div>
    </div>
  );
}
