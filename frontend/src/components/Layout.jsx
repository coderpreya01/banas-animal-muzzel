import { NavLink, Outlet } from "react-router-dom";

const links = [
  // ["/", "🏠 Dashboard"],
  // ["/register", "📝 Register Animal"],
  ["/identify", "🎬 Identify from Video"],
  // ["/verify", "✅ Verify / Search"],
  // ["/live", "📡 Live Camera"],
  // ["/records", "📋 Animal Records"],
];

export default function Layout() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">Muzzle Re-ID</div>
        <nav>
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
