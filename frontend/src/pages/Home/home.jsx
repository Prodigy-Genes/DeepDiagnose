
import './Home.css';
import Header from '../../components/home-page/header/header';
import Footer from '../../components/home-page/footer/footer';
import Body from './Body';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="page-container">
      <Header />
      <main className="content">
        {/* your page’s main content goes here */}
        <Body />
      </main>
      <Footer />

      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 right-4">
          <Link 
            to="/debug"
            className="bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-gray-700 transition"
          >
            🛠️ Debug Tools
          </Link>
        </div>
      )}
    </div>
    
  );
}
