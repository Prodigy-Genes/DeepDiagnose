
import './Home.css';
import Header from '../../components/home-page/header/header';
import Footer from '../../components/home-page/footer/footer';
import Body from './Body';
export default function Home() {
  return (
    <div className="page-container">
      <Header />
      <main className="content">
        {/* your page’s main content goes here */}
        <Body />
      </main>
      <Footer />
    </div>
    
  );
}
