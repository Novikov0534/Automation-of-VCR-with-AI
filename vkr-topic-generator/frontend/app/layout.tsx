import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VKR AI — генератор тем ВКР",
  description: "Интеллектуальное формирование тем ВКР для 09.03.01",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ru"><body>{children}</body></html>;
}
