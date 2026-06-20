export default function FederatedPage() {
  return (
    <div className="min-h-screen bg-base-950 text-zinc-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-semibold mb-4">Federated Learning</h1>
        <p className="text-sm text-zinc-400 mb-6">
          This page will show federated node status, model synchronization, and training progress.
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
