import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Register from "./pages/Register.jsx";
import IdentifyVideo from "./pages/IdentifyVideo.jsx";
// import Verify from "./pages/Verify.jsx";
// import LiveCamera from "./pages/LiveCamera.jsx";
// import Records from "./pages/Records.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        {/* <Route path="register" element={<Register />} /> */}
        <Route path="identify" element={<IdentifyVideo />} />
        {/* <Route path="verify" element={<Verify />} /> */}
        {/* <Route path="live" element={<LiveCamera />} /> */}
        {/* <Route path="records" element={<Records />} /> */}
      </Route>
    </Routes>
  );
}
