import '../styles/globals.css'
import { ProSidebarProvider } from 'react-pro-sidebar';
function MyApp({ Component, pageProps }) {
  return (
    <Component {...pageProps} />
  )
}

export default MyApp
