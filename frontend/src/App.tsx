import { useRoute } from "./router";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Analyze } from "./pages/Analyze";
import { HowItWorks } from "./pages/HowItWorks";
import { About } from "./pages/About";
import { Insights } from "./pages/Insights";

export default function App() {
  const route = useRoute();

  let page;
  switch (route) {
    case "/analyze":
      page = <Analyze />;
      break;
    case "/how-it-works":
      page = <HowItWorks />;
      break;
    case "/about":
      page = <About />;
      break;
    case "/insights":
      page = <Insights />;
      break;
    default:
      page = <Home />;
  }

  return <Layout route={route}>{page}</Layout>;
}
